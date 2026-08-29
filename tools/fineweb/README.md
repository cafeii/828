# FineWeb 下载与加载

数据源：ModelScope `swift/fineweb`（HF `HuggingFaceFW/fineweb` 镜像）。
下载子集：`sample/10BT`（15 文件 / 30.7GB，用于 1B 模型）与 `sample/100BT`（150 文件 / 303.2GB，用于 300M 模型）。

## 流程

1. **本地生成清单**：`python3 enumerate.py` → `manifest.json`（固定文件 id，已提交 git）。
2. **服务器下载**：`python3 download.py [manifest] [目标目录] [并行数]`。
   默认目标 `/work/projects/memos-b3/datasets/lzc_rnn/fineweb/`。
   中断后重跑同一命令即可按文件续传（已完成的文件按大小校验跳过）。
3. **lit_gpt 加载**：数据落盘为 HF 原版 parquet 布局，用
   `scripts/data/prepare_fineweb.py`（适配自 litgpt `prepare_slimpajama.py`）
   转成 litdata 预分块格式后即可 `pretrain`。
   FineWeb 无官方 validation/test，需要时从子集尾部挪 1–2 个 parquet 单独 prepare 一份。

备注：ModelScope 上 `swift/SlimPajama-627B` 仅有 test 集（0.02GB），
`xpengx/SlimPajama-627B` 为 204 卷 7z（957GB，不可采样解压），故改用 FineWeb。
