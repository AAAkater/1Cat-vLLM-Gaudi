# SPDX-License-Identifier: Apache-2.0
"""Build an isolated eight-rank stock RS/AG native HCL from pinned TP4 objects."""
import argparse
import concurrent.futures
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Source contract differs ({text.count(old)} != {count}): {old}")
    return text.replace(old, new)


def transform(name, text):
    if name.endswith('/collectives/all_reduce.cpp'):
        text = replace(text, 'if (comm.getCommSize() != 4 || comm.getScaleupGroupSize() != 4 || phase < 0 || phase > 2 ||',
                       'const auto ranks = comm.getCommSize();\n    if ((ranks != 4 && ranks != 8) || comm.getScaleupGroupSize() != ranks || phase < 0 || phase > 2 ||')
        text = replace(text, 'params.m_count % 4 != 0', 'params.m_count % ranks != 0')
        text = replace(text, 'params.m_count / 4', 'params.m_count / ranks', 2)
    elif name.endswith('/hccl/hccl.cpp'):
        text = replace(text, 'extern "C" uint32_t HCCL_API_CALL hcclTp4NativeGraphVersion() { return 1; }',
                       'extern "C" uint32_t HCCL_API_CALL hcclTp4NativeGraphVersion() { return 1; }\nextern "C" uint32_t HCCL_API_CALL hcclTp8NativeGraphVersion() { return 1; }')
        text = replace(text, '(phase != 2 && count % 4 != 0) || ', '')
        text = replace(text, 'communicator->getCommSize() != 4 ||\n        communicator->getDynamicComm()->getScaleupGroupSize() != 4) return hcclInvalidArgument;',
                       '(communicator->getCommSize() != 4 && communicator->getCommSize() != 8) ||\n        communicator->getDynamicComm()->getScaleupGroupSize() != communicator->getCommSize() ||\n        (phase != 2 && count % communicator->getCommSize() != 0)) return hcclInvalidArgument;')
    elif name.endswith('/gaudi2/context_manager.cpp'):
        text = replace(text, 'if (m_device.getComm(comm).getCommSize() == 4)',
                       'if (m_device.getComm(comm).getCommSize() == 4 || m_device.getComm(comm).getCommSize() == 8)')
    elif name.endswith('/gen2_arch_common/hcl_collective_routines.cpp'):
        text = replace(text, '(ranks == 2 || ranks == 4)', '(ranks == 2 || ranks == 4 || ranks == 8)')
        text = replace(text, 'VERIFY(ranks == 4 ? capture->tp4Phase', 'VERIFY(ranks > 2 ? capture->tp4Phase')
    else:
        raise ValueError(name)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    baseline, output = args.baseline_build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(subprocess.check_output(['ninja', '-C', str(baseline), '-t', 'compdb'], text=True))
    suffixes = ('/collectives/all_reduce.cpp', '/hccl/hccl.cpp', '/gaudi2/context_manager.cpp',
                '/gen2_arch_common/hcl_collective_routines.cpp')
    selected = [row for row in rows if row['file'].endswith(suffixes)]
    assert len(selected) == 4, [r['file'] for r in selected]
    proof = {'baseline_binary': {'path': str(baseline/'lib/libhcl.so'), 'sha256': digest(baseline/'lib/libhcl.so')},
             'source': [], 'objects': [], 'status': 'building'}
    replacements = {}
    commands = []
    for index, row in enumerate(selected):
        source = Path(row['file']); original = source.read_text(); updated = transform(str(source), original)
        private = output/f'{index}-{source.name}'; private.write_text(updated)
        (output/f'{index}.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True), fromfile=str(source), tofile=str(private))))
        command = shlex.split(row['command'])
        old_object = command[command.index('-o')+1]
        new_object = output/f'{index}.o'
        for flag, value in (('-o', str(new_object)), ('-MF', str(output/f'{index}.d')), ('-MT', str(new_object))):
            command[command.index(flag)+1] = value
        command[command.index('-c')+1] = str(private)
        # Quoted headers relative to the original source continue to resolve.
        command.insert(1, f'-I{source.parent}')
        replacements[old_object] = str(new_object)
        proof['source'].append({'path': str(source), 'sha256': digest(source), 'private': str(private), 'private_sha256': digest(private)})
        commands.append(command)
    raw_link = subprocess.check_output(['ninja', '-C', str(baseline), '-t', 'commands', 'lib/libhcl.so'], text=True).splitlines()[-1]
    link = shlex.split(raw_link)
    assert link[:2] == [':', '&&'], link[:2]
    link = link[2:link.index('&&', 2)]
    link[link.index('-o')+1] = str(output/'libhcl.so')
    for index, arg in enumerate(link):
        if arg in replacements:
            link[index] = replacements[arg]
        elif arg.endswith(('.o', '.a', '.so')):
            path = Path(arg) if Path(arg).is_absolute() else baseline/arg
            if path.is_file():
                link[index] = str(path)
                proof['objects'].append({'path': str(path), 'sha256': digest(path)})
    proof['commands'] = {'compile': commands, 'link': link}
    manifest = output/'BUILD_PROOF.json'; manifest.write_text(json.dumps(proof, indent=2)+'\n')
    def compile_one(pair):
        index, command = pair
        with (output/f'compile-{index}.log').open('w') as log:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(compile_one, enumerate(commands)))
    with (output/'link.log').open('w') as log:
        subprocess.run(link, cwd=baseline, stdout=log, stderr=subprocess.STDOUT, check=True)
    for record in proof['source']+proof['objects']:
        assert digest(Path(record['path'])) == record['sha256'], record['path']
    assert digest(baseline/'lib/libhcl.so') == proof['baseline_binary']['sha256']
    symbols = subprocess.check_output(['nm', '-D', str(output/'libhcl.so')], text=True)
    assert ' hcclTp8NativeGraphVersion' in symbols
    proof.update(status='built', binary_sha256=digest(output/'libhcl.so'))
    manifest.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({'status': proof['status'], 'binary': str(output/'libhcl.so'), 'sha256': proof['binary_sha256']}), flush=True)


if __name__ == '__main__':
    main()
