# Kernel import shim：统一从 fla 0.6.0 公开 API 取 Triton kernel。
#
# 历史：最初 import third_party/GatedDeltaNet-2 vendored 的 gdn2_ops，但那份 kernel
# 调用 fla 内部函数且锚定主干未发布窗口（use_exp2/transpose_state_layout 共存期），
# 与 fla 0.6.0 不兼容。fla 0.6.0 已上游化完整的 fla.ops.gdn2（签名一致，
# 仅 transpose_state_layout 改名 state_v_first，本项目不使用该参数），
# 故改用 fla 公开 API（详见 patches/PATCHES.md）。
#
# 惰性加载：本地CPU环境无 triton/fla，本模块可 import，get_* 首次调用时才触发依赖。


def get_chunk_gdn2():
    from fla.ops.gdn2 import chunk_gdn2

    return chunk_gdn2


def get_fused_recurrent_gdn2():
    from fla.ops.gdn2 import fused_recurrent_gdn2

    return fused_recurrent_gdn2


def get_chunk_kda():
    from fla.ops.kda import chunk_kda

    return chunk_kda


def get_fused_recurrent_kda():
    from fla.ops.kda import fused_recurrent_kda

    return fused_recurrent_kda


def get_chunk_gated_delta_rule():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def get_fused_recurrent_gated_delta_rule():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule
