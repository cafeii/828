# RULER 本地改动记录

vendored from https://github.com/NVIDIA/RULER @ c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a

改动目的：接入工作区自训 lit_gpt 模型（GQA baseline / LSA），并去掉 nemo-toolkit 硬依赖。
评测口径见 `docs/experiment.md`（S-NIAH-1/2/3 + MK-NIAH-1）与 `plans/2026-09-01-lsa300m-eval.md`（1K/2K/4K/8K）。

## 1. 新增 `scripts/pred/litgpt_model.py`（新文件）

`LitGPTModel` 类，形态仿同目录 `model_wrappers.py` 的 `MambaModel`，实现 RULER 要求的
`__call__(prompt) -> {'text': [str]}` 与 `process_batch(prompts) -> List[dict]` 两个方法。
内部复用 `scripts/eval/wrapper.py:load_eval_model`，不重复实现加载逻辑。

要点：
- `name_or_path` 按 `"<model_name>:<ckpt_path>"` 约定编码（RULER 只透传单个 MODEL_PATH 字段）。
- 本模型不支持 padding mask（`wrapper.py` 中对非全 1 mask 直接 raise），故 `process_batch` 逐条
  循环，且 RULER 侧 `BATCH_SIZE` 必须为 1。
- `temperature=0` 时显式设 `do_sample=False` 并去掉 top_p/top_k，对齐 RULER 的贪心口径。

## 2. `scripts/pred/call_api.py`

- 第 45 行 `from nemo.collections.asr.parts.utils.manifest_utils import read_manifest`
  → 改为 `sys.path.append(父目录)` + `from data.manifest_utils import read_manifest`。
- `SERVER_TYPES` 元组增加 `'litgpt'`。
- `get_llm()` 在 `mamba` 分支前增加 `litgpt` 分支，构造 `LitGPTModel`。

## 3. `scripts/eval/evaluate.py`

- 第 41 行同款 nemo import → 改用 `data.manifest_utils`（`read_manifest, write_manifest`）。

## 4. `scripts/data/manifest_utils.py`

- 上游此文件只有 `write_manifest`，`read_manifest` 原本依赖 nemo。补一个等价的纯 json 实现
  （逐行 `json.loads`，跳过空行），使 pred/eval 全链路无 nemo 依赖。

## 5. `scripts/data/prepare.py`

- 第 124 行 `command = f"python {script} ..."` → `f"{sys.executable} {script} ..."`（补 `import sys`）。
  uv venv 无 `python` 别名时子进程静默失败。
- 子进程失败的 `except CalledProcessError` 原本只 print 不抛出，上层照常打印
  "Prepare ... with lines: N"，批量跑时会伪装成成功产出空/残缺数据 → 补 `raise`。

## 6. `scripts/data/synthetic/niah.py`（样本长度达标修复）

- 生成阶段超限回退原为 `used_haystack -= incremental`（essay=500 词/步）：一次超限即砍掉
  ~650 token 的 haystack，实测 1K 样本仅 ~190 token（利用率 19%）。改为逐 1 回退 + 下界保护。
- 注意 essay 的 `incremental=500` **不可调小**：`tokens_per_haystack` 用 incremental 词的样本估计，
  分母太小时模板 token 占比过大 → 高估每词 token → `upper_bound` 被低估 → 二分搜索够不到
  目标长度（incremental=10 实测 1K 利用率仅 60%）。保持上游原值。
- 修复后 1K/2K/4K/8K 四档实测样本总长利用率均 100%（误差 <0.5%）。

## 7. 配置（`scripts/config_models.sh` / `config_tasks.sh`）

不改上游文件，改由工作区脚本 `scripts/eval/run_ruler.sh` 直接驱动 `prepare.py` / `call_api.py` /
`evaluate.py` 三段，并在其中固定：4 任务子集、SEQ_LENGTHS=(1024 2048 4096 8192)、
`MODEL_TEMPLATE_TYPE=base`、tokenizer 显式走 hf（避免 `config_models.sh` 见到
`tokenizer.model` 就误判为 nemo SentencePiece 分支）、`BATCH_SIZE=1`。
