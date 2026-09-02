# prefix-linear-attention 补丁记录

对本第三方库的所有改动登记在此。原则：只新增文件，尽量不改已有文件；必须改时保持最小 diff 并在此说明。

## 1. 新增 `lm-eval-harness/lm_eval/models/litgpt_lm.py`

- **目的**：把工作区自训 lit_gpt 模型（fabric ckpt）接入本 harness，用于 JRT 召回类评估（`based_*` 任务）。
- **内容**：注册名 `litgpt` 的 `LitGPTLM(HFLM)`，仿照同目录 `local_lm.py` 的做法——把已构造好的 model+tokenizer 传给 `HFLM.__init__(pretrained=model, backend="causal", tokenizer=tokenizer, ...)`。模型经 `<workspace>/scripts/eval/wrapper.py` 的 `load_eval_model()` 加载（`sys.path` 由本文件位置向上 5 级定位 workspace 根）。
- **model_args**：`model_name, ckpt_path, tokenizer_path, max_length=4096, device=cuda, dtype=bfloat16`。
- **约束**：本模型 packed 定长训练、不支持 padding mask，生成类任务须 `--batch_size 1`。

## 2. 修改 `lm-eval-harness/lm_eval/models/__init__.py`（1 行）

- **目的**：让上面的模型类在 import `lm_eval.models` 时完成注册（该 fork 的模型注册依赖此处显式 import）。
- **diff**：文件末尾 `from . import jrt_lm` 之后新增一行
  ```python
  from . import litgpt_lm  # PATCH(rnn工作区): 自训lit_gpt模型接入，见 ../patches/PATCHES.md
  ```
- 未改动其他任何已有文件。

## 备注：任务可用性（非改动，排查结论）

本 fork 的 `lm_eval/tasks/` 只含 `based_*`（JRT 12 项）、`scrolls`、`super_glue`（含 boolq）。
标准零样本任务中 **只有 BoolQ 在本 fork 内**；Lambada/PIQA/Hellaswag/Winogrande/ARC-e/ARC-c/OpenBookQA/SIQA 均不在，
需另行安排（见工作区 plans/ 中的评估计划）。故本目录下的脚本只承担 JRT 部分：`scripts/eval/run_jrt.sh`。
