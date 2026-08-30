# Triton kernel import shim：以包形式惰性加载
# third_party/GatedDeltaNet-2/lit_gpt/gdn2_ops 下的 kernel（零改动、零复制第三方代码）。
# gdn2_ops 无 __init__.py 且内部有相对导入（from .chunk_kda import ...），
# 因此先用 importlib 把 'gdn2_ops' 注册为 namespace package 再导入子模块。
# 惰性加载：本地CPU环境无 triton/fla，本模块可 import，get_* 首次调用时才触发依赖。

import importlib
import importlib.util
import sys
from pathlib import Path

_GDN2_OPS_DIR = (
    Path(__file__).resolve().parents[2]
    / "third_party" / "GatedDeltaNet-2" / "lit_gpt" / "gdn2_ops"
)

_cache = {}


def _ensure_fla_compat():
    """gdn2_ops 针对更新版 fla 编写；对旧版 fla（0.6.0）缺失的符号做运行时注入兜底。"""
    import os

    import fla.utils

    if not hasattr(fla.utils, "USE_CUDA_GRAPH"):
        fla.utils.USE_CUDA_GRAPH = os.getenv("FLA_USE_CUDA_GRAPH", "0") == "1"


def _load(module_name: str, attr: str):
    key = (module_name, attr)
    if key not in _cache:
        _ensure_fla_compat()
        if "gdn2_ops" not in sys.modules:
            spec = importlib.machinery.ModuleSpec("gdn2_ops", None, is_package=True)
            spec.submodule_search_locations = [str(_GDN2_OPS_DIR)]
            sys.modules["gdn2_ops"] = importlib.util.module_from_spec(spec)
        mod = importlib.import_module(f"gdn2_ops.{module_name}")
        _cache[key] = getattr(mod, attr)
    return _cache[key]


def get_chunk_gdn2():
    return _load("chunk_gdn2", "chunk_gdn2")


def get_fused_recurrent_gdn2():
    return _load("fused_recurrent_gdn2", "fused_recurrent_gdn2")


def get_chunk_kda():
    return _load("chunk_kda", "chunk_kda")
