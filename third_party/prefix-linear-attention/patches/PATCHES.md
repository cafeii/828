# prefix-linear-attention 补丁记录

对本第三方库的所有改动登记在此。原则：只新增文件，尽量不改已有文件；必须改时保持最小 diff 并在此说明。

## 1. 新增 `lm-eval-harness/lm_eval/models/litgpt_lm.py`

- **目的**：把工作区自训 lit_gpt 模型（fabric ckpt）接入本 harness，用于 JRT 召回类评估（`based_*` 任务）。
- **内容**：注册名 `litgpt` 的 `LitGPTLM(HFLM)`，仿照同目录 `local_lm.py` 的做法——把已构造好的 model+tokenizer 传给 `HFLM.__init__(pretrained=model, backend="causal", tokenizer=tokenizer, ...)`。模型经 `<workspace>/scripts/eval/wrapper.py` 的 `load_eval_model()` 加载（`sys.path` 由本文件位置向上 5 级定位 workspace 根）。
- **model_args**：`model_name, ckpt_path, tokenizer_path, max_length=4096, device=cuda, dtype=bfloat16`。
- **约束**：本模型 packed 定长训练、不支持 padding mask，生成类任务须 `--batch_size 1`。

## 2. 修改 `lm-eval-harness/lm_eval/models/__init__.py`（最小 diff）

- **目的**：让上面的模型类在 import `lm_eval.models` 时完成注册（该 fork 的模型注册依赖此处显式 import）。
- **diff**：
  1. 末尾新增 `from . import litgpt_lm`（自训模型接入）。
  2. `based_lm` / `jrt_lm` / `local_lm` 三个 import 移入 try/except ImportError 守卫——它们依赖
     `based` / `train` / `hydra` 等本工作区不需要的包，在 lzc-rnn 环境会阻塞整个 models 包的
     import（2026-09-02 服务器实测）。guard 后未注册基于这些依赖的模型，但不影响 litgpt/hf。
- 未改动其他任何已有文件。

## 备注：任务可用性（非改动，排查结论）

本 fork 的 `lm_eval/tasks/` 只含 `based_*`（JRT 12 项）、`scrolls`、`super_glue`（含 boolq）。
标准零样本任务中 **只有 BoolQ 在本 fork 内**；Lambada/PIQA/Hellaswag/Winogrande/ARC-e/ARC-c/OpenBookQA/SIQA 均不在，
需另行安排（见工作区 plans/ 中的评估计划）。故本目录下的脚本只承担 JRT 部分：`scripts/eval/run_jrt.sh`。

## 命名说明（2026-09-02）

工作区架构命名口径由 LSA 统一为 LSR（Latent State RNN），`litgpt_lm.py` 注释中的示例 model_name
同步为 `gdn2_lsr_340M`。模型注册名 `litgpt` 不变。

## 3. 修改 `lm-eval-harness/lm_eval/api/task.py`（transformers 5.x 兼容，2 处）

- `truncate_context` 里的两处 `tokenizer.batch_encode_plus(...)` 改为等价的
  `tokenizer(...)`（`__call__`）。`batch_encode_plus` 在 transformers 5.x 已移除
  （lzc-rnn 为 5.16.1，JRT 冒烟 2026-09-02 实测触发 AttributeError）。
- 语义不变：单条文本、相同参数（return_tensors/pt/padding/truncation），输出同一 input_ids。
