# 训练环境说明

- **本地（mac，开发）**：根目录 uv project（`pyproject.toml` + `uv.lock`），
  `uv sync` 一键复现（python 3.11，CPU/MPS 依赖子集）。
- **服务器（训练）**：`lzc-rnn` conda env，由 `build_env.sh` 幂等构建：
  1. `requirements-lock.txt` —— `uv export` 产物（本地更新依赖后需重新导出：
     `uv export --format requirements-txt --no-hashes -o scripts/setup/requirements-lock.txt`）
  2. `flash-attn==2.8.3` —— 预编译 wheel 优先，源码编译兜底
  3. `fla` —— editable 安装自 `third_party/flash-linear-attention`

依赖范围对齐 `third_party/GatedDeltaNet-2/requirements.txt`（minimal）+ 数据处理件
（litdata/pyarrow/zstandard）。mamba/tilelang/torchdata-nightly 等基线实验件未包含，需要时再补。

注意：`setuptools<81` 是 lightning 2.1.2 的硬要求（pkg_resources 在 81+ 被移除）。
