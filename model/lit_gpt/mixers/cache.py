# 推理用 per-layer 状态缓存：与 fla.models.utils.Cache 的存取协议兼容
# （len/下标返回 per-layer dict；update(layer_idx=..., **states) 写入并返回该 dict），
# 但不依赖 fla，CPU 环境（无 triton）也可用。GPU 评估如需 transformers
# GenerationMixin 集成，可直接换用 fla.models.utils.Cache，mixer 侧无感知。

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SimpleCache:
    def __init__(self) -> None:
        self.layers: List[Dict[str, Any]] = []
        self.seen_tokens: int = 0

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        return self.layers[layer_idx]

    def update(self, layer_idx: int = 0, offset: int = 1, **states: Any) -> Dict[str, Any]:
        while len(self.layers) <= layer_idx:
            self.layers.append({})
        layer = self.layers[layer_idx]
        # 显式传入的键一律写入（含None）：mixer侧用下标直接访问（如last_state["conv_state"]），
        # 键必须始终存在，与fla Cache行为一致
        layer.update(states)
        if layer_idx == 0:
            self.seen_tokens += offset
        return layer


def _require_layer_idx(module) -> int:
    layer_idx = getattr(module, "layer_idx", None)
    assert layer_idx is not None, f"{type(module).__name__} 未设置 layer_idx，无法使用推理缓存"
    return layer_idx


def get_layer_cache(module, past_key_values) -> Optional[Dict[str, Any]]:
    if past_key_values is None:
        return None
    layer_idx = _require_layer_idx(module)
    if len(past_key_values) > layer_idx:
        return past_key_values[layer_idx]
    return None


def update_layer_cache(module, past_key_values, **kwargs) -> Optional[Dict[str, Any]]:
    if past_key_values is None:
        return None
    return past_key_values.update(layer_idx=_require_layer_idx(module), **kwargs)
