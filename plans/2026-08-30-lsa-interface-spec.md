# model/lit_gpt 接口规格（subagent开发约定）

本文件是本期开发的接口契约，所有文件以此为准。总计划见 2026-08-30-lsa-litgpt-dev.md。

## 目录布局

```
model/
  lit_gpt/
    __init__.py          # re-export GPT, Block, Config, FusedCrossEntropyLoss
    config.py            # [主agent写] Config dataclass + 各规模/方案配置
    model.py             # [subagent A] GPT/Block/CausalSelfAttention/LLaMAMLP
    mixers/
      __init__.py        # [主agent写]
      gdn2.py            # [主agent写] GatedDeltaNet2层（GQA/LSA/GVA/expand_v统一）
      naive.py           # [subagent B] 纯torch逐token递归参考实现（CPU可跑）
    kernels.py           # [subagent A] third_party kernel import shim（惰性）
    rmsnorm.py           # [subagent A] 拷贝自GDN2
    fused_cross_entropy.py / packed_dataset.py / speed_monitor.py /
    utils.py / tokenizer.py / rotary.py / fused_rotary_embedding.py
                         # [subagent A] 拷贝自GDN2，仅改内部import路径
tests/
  test_lsa_equivalence.py   # [subagent B] 本地CPU数学单测
  test_kernel_parity.py     # [后续] 服务器GPU
scripts/
  pretrain.py               # [Phase 2]
```

## 关键约束

- 本地mac无triton/fla/flash_attn：`kernels.py`、`mixers/gdn2.py`、`model.py`中所有
  fla/flash_attn/triton相关import必须惰性（函数/构造器内import或try-except守卫），
  保证 `from lit_gpt.config import Config` 和 `mixers.naive` 在纯CPU环境可import。
- third_party代码零改动零复制：`kernels.py`把
  `third_party/GatedDeltaNet-2/lit_gpt`（gdn2_ops所在包）加入sys.path后
  re-export `chunk_gdn2`、`fused_recurrent_gdn2`、`chunk_kda`。
  注意gdn2_ops内部有相对导入（`from .chunk_kda import ...`），shim需以包形式导入
  （importlib加载`gdn2_ops`为顶层包，包搜索路径指向third_party目录）。
- 代码风格：简洁易读，与GDN2原实现对齐；不做实验设计外的功能。

## Config 新增字段（dataclass，主agent实现，此处为契约）

```python
mixer: str = "attn"            # "attn" | "gdn2"（gdn/kda后续加）
mixer_per_layer: int = 1       # 1=全RNN层；N>1=每N层1个RNN其余attn；-1=纯attn
num_groups: Optional[int] = None   # RNN组数G；None→n_head（即MHA形态）
head_dim: Optional[int] = None     # RNN头维d_k；None→n_embd//n_head
expand_v: float = 1.0              # value头维扩展（方案：GQA+expand v_dim）
num_v_heads: Optional[int] = None  # 组内v头数>1（方案：GQA+增加v_head）；None→等同组数
use_lsa: bool = False              # LSA开关
lsa_latent_dim: Optional[int] = None  # d_c；None→head_v_dim
use_short_conv: bool = True
conv_size: int = 4
allow_neg_eigval: bool = False
nope: bool = True                  # RNN实验默认无位置编码
```

沿用原字段：n_layer/n_head/n_embd/intermediate_size/_norm_class/_mlp_class/
mamba_init/local_window/bias/parallel_residual等。

## mixers/gdn2.py 层接口（主agent实现，契约）

```python
class GatedDeltaNet2(nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups=None, head_dim=None,
                 expand_v=1.0, num_v_heads=None, use_lsa=False, lsa_latent_dim=None,
                 mode="chunk", use_short_conv=True, allow_neg_eigval=False,
                 conv_size=4, conv_bias=False, layer_idx=None, norm_eps=1e-5): ...
    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, **kwargs) -> tuple[o, None, past_key_values]
```

语义（G=num_groups，I=num_heads//G，d_k=head_dim，d_v=head_dim*expand_v，d_c=lsa_latent_dim）：
- q per-head：`q_proj: hidden→H*d_k`
- k/擦除门b/遗忘门g 组级：`G*d_k`；写门w与潜v 组级：`G*d_c`（LSA）或 `G*d_v`（GQA）
- LSA=True：v_latent `G*d_c`，P参数 `[H, d_v, d_c]`，kernel后
  `einsum('bthc,hvc->bthv', o, P)`；LSA=False且G<H即GQA（组级v直接被所有组内q头读取）
- kernel调用（策略2）：组级张量repeat至H份进`chunk_gdn2`，q不repeat
- 输出门g_proj per-head `H*d_v`，o_norm(d_v)，o_proj `H*d_v→hidden`
- A_log/dt_bias形状按组级门（G*d_k相关）设置，repeat进kernel

## mixers/naive.py（subagent B实现，契约）

纯torch（无einops外部依赖亦可用einops，无fla/triton），fp32，CPU可跑：

```python
def naive_gdn2_recurrence(q, k, v, g, b, w, scale=None, initial_state=None):
    """逐token递归GDN2参考实现。
    q:[B,T,H,dk] k:[B,T,H,dk] v:[B,T,H,dv] g:[B,T,H,dk](log-decay)
    b:[B,T,H,dk] w:[B,T,H,dv]
    S_t = (I - k b⊙k^T) Diag(exp(g)) S_{t-1} + k (w⊙v)^T ; o_t = scale * q_t S_t
    q/k先做L2 norm（与kernel的use_qk_l2norm_in_kernel=True对齐）。
    返回 (o:[B,T,H,dv], S:[B,H,dk,dv])"""

def naive_lsa_forward(q, k, c, g, b, w, P, scale=None):
    """LSA潜空间递归：组级T_g递归 + 出口乘P。
    q:[B,T,H,dk] k/g/b:[B,T,G,dk] c:[B,T,G,dc] w:[B,T,G,dc] P:[H,dv,dc]
    组级张量内部repeat到H后走naive_gdn2_recurrence，再乘P。返回o:[B,T,H,dv]"""

def naive_lsa_expanded_forward(q, k, c, g, b, w, P, scale=None):
    """策略1参考：入口展开 v_{g,i}=P_{g,i}(w⊙c)，per-head真实state递归（w口传1）。
    用于验证与naive_lsa_forward等价。"""
```

## tests/test_lsa_equivalence.py（subagent B实现）

pytest，CPU，fp64/fp32小尺寸（如B=2,T=8,H=8,G=2,dk=16,dv=16,dc=16）：
1. `naive_lsa_forward` ≈ `naive_lsa_expanded_forward`（P可提出性，atol~1e-5）
2. P为分块结构 `P_{g,i}=I`（dc=dv）时，LSA输出 == GQA（组级v直读）输出
3. 状态形状断言：潜state为G份
4. 门语义：w=1,b与GDN2一致时递归与手写einsum一步展开一致

运行方式：`uv run pytest tests/test_lsa_equivalence.py`（repo根）。
