# SPDX-License-Identifier: Apache-2.0
"""Guard the HPU DSpark adapter against upstream speculator field drift.

``HPUDSparkSpeculator`` deliberately skips ``DSparkSpeculator.__init__`` (it
needs a GPU ``VllmConfig``) and assigns only the fields the inherited upstream
loop reads.  If upstream starts reading a new ``self.<field>`` in an inherited
method, this test fails instead of the HPU path raising at serving time.
"""
import ast
import inspect
import textwrap
from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_gaudi.v1.spec_decode.deepseek_v41_dspark import HPUDSparkSpeculator

_INHERITED_ENTRY = "_generate_draft"


def _self_attributes(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }


def _reachable_inherited_attributes():
    """Collect self fields read by every inherited method reachable from the entry."""
    pending, seen, fields = [_INHERITED_ENTRY], set(), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in HPUDSparkSpeculator.__dict__:
            continue  # HPU override: its own reads are covered by its own tests.
        function = getattr(DSparkSpeculator, name, None)
        if not callable(function):
            continue
        for attribute in _self_attributes(function):
            if callable(getattr(DSparkSpeculator, attribute, None)):
                pending.append(attribute)
            else:
                fields.add(attribute)
    return fields


class _Draft(torch.nn.Module):

    def __init__(self):
        super().__init__()
        last = SimpleNamespace(
            markov_head=SimpleNamespace(embed=torch.zeros(4, 2), head=SimpleNamespace(weight=torch.zeros(4, 2))),
            confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1, 4))),
        )
        self.weights = SimpleNamespace(get_submodule=lambda name: last)
        self.layers = ()


def test_inherited_upstream_loop_reads_only_fields_the_hpu_adapter_sets():
    speculator = HPUDSparkSpeculator(_Draft(), device="cpu", max_num_reqs=2)
    fields = _reachable_inherited_attributes()
    assert "draft_tokens" in fields, "upstream loop structure changed; revisit the HPU adapter"
    missing = sorted(field for field in fields if not hasattr(speculator, field))
    assert not missing, f"upstream DSpark loop now reads fields the HPU adapter does not set: {missing}"
