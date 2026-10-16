# SPDX-License-Identifier: Apache-2.0
"""Read global Synapse HBM counters without acquiring or synchronizing a device.

Call only at an existing worker lifecycle boundary, outside captured/compiled
model code. These counters differ from torch.hpu.mem_get_info(), which reports
the Bridge allocator pool. This helper deliberately imports no Torch module,
loads no new Synapse runtime and never changes device selection.
"""

import ctypes
from functools import lru_cache
import operator
import os
from pathlib import Path
import sys
import time


def _synapse_paths(maps):
    paths = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5]
        name = Path(path).name
        if name == "libSynapse.so" or name.startswith("libSynapse.so."):
            if not path.endswith(" (deleted)"):
                paths.add(path)
    return paths


@lru_cache(maxsize=1)
def _loaded_synapse_library():
    # Refuse ambiguity, missing NOLOAD support, and an unloaded SDK. CDLL's
    # default lookup could introduce an unintended second runtime instance.
    paths = _synapse_paths(Path("/proc/self/maps").read_text())
    if len(paths) != 1:
        raise RuntimeError("Expected one already-loaded Synapse library")
    no_load = getattr(os, "RTLD_NOLOAD", None)
    if no_load is None:
        raise RuntimeError("RTLD_NOLOAD is unavailable")
    return ctypes.CDLL(next(iter(paths)), mode=no_load | ctypes.RTLD_LOCAL)


def read_synapse_memory():
    """Return a JSON-safe best-effort snapshot; never initialize an HPU.

    The installed Gaudi Bridge binds hpu.current_device() directly to
    HPUDeviceContext::get_device().id(), the acquired Synapse device ID.
    Do not substitute torch.local_rank, HLS_MODULE_ID, or a hardcoded zero.
    The caller owns any desired synchronization; this is a non-atomic read.
    """
    record = {
        "status": "unavailable",
        "scope": "synapse_global_device_memory",
        "synchronized_by_helper": False,
        "timestamp_ns": time.time_ns(),
        "device_id": None,
        "device_id_source": "torch.hpu.current_device (Bridge acquired Synapse ID)",
        "synapse_status": None,
        "free_bytes": None,
        "total_bytes": None,
    }
    stage = "torch_state"
    try:
        torch = sys.modules.get("torch")
        hpu = getattr(torch, "hpu", None)
        initialized = getattr(hpu, "is_initialized", None)
        if initialized is None:
            record["reason"] = "torch_hpu_not_loaded"
            return record
        if not initialized():
            record.update(status="not_initialized", reason="hpu_not_initialized")
            return record
        stage = "device_id"
        device = hpu.current_device()
        if isinstance(device, bool):
            raise ValueError("Boolean device ID")
        device = operator.index(device)
        if not 0 <= device <= 0xFFFFFFFF:
            raise ValueError("Device ID outside synDeviceId range")
        record["device_id"] = device
        stage = "loaded_runtime"
        api = _loaded_synapse_library()
        query = api.synDeviceGetMemoryInfo
        query.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int
        free, total = ctypes.c_uint64(), ctypes.c_uint64()
        stage = "memory_query"
        status = int(query(device, ctypes.byref(free), ctypes.byref(total)))
        record["synapse_status"] = status
        if status != 0:
            record.update(status="synapse_error", reason="synDeviceGetMemoryInfo_failed")
        elif total.value == 0 or free.value > total.value:
            record.update(status="invalid_result", reason="invalid_global_memory_range")
        else:
            record.update(status="ok", free_bytes=free.value, total_bytes=total.value)
    except Exception as error:
        # Preserve the failure class/stage without dumping process paths,
        # environment, credentials, or arbitrary exception messages to logs.
        record.update(status="error", stage=stage, error_type=type(error).__name__)
    return record
