# third_party

外部资源备案。下列仓库已清除 `.git` 元数据（vendor 化），如需升级请对照下表重新 clone 指定 commit。

## 代码仓库

| 目录 | 上游 | commit | 提交日期 |
|---|---|---|---|
| GatedDeltaNet | https://github.com/NVlabs/GatedDeltaNet | b53d6d3a161267432a79c1c04af69fa52bddc921 | 2026-03-13 |
| GatedDeltaNet-2 | https://github.com/NVlabs/GatedDeltaNet-2 | 95709fc250357c2dd109361c353192f2aa5913f9 | 2026-05-25 |
| flash-linear-attention | https://github.com/fla-org/flash-linear-attention | 6fa5016f5aae3a54080813ad70f149466a8f5b86 | 2026-08-28 |
| prefix-linear-attention | https://github.com/HazyResearch/prefix-linear-attention | 7490f22bda3e38dc057bcbaffe6bdb09b4d475e6 | 2024-07-08 |
| LongBench | https://github.com/THUDM/LongBench | 2e00731f8d0bff23dc4325161044d0ed8af94c1e | 2025-01-15 |
| RULER | https://github.com/NVIDIA/RULER | c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a | 2026-07-22 |

备注：prefix-linear-attention 依赖 ThunderKittens 子模块（https://github.com/HazyResearch/ThunderKittens @ 2b9827dc11be408c386e70b84f923a11f70c7c33），vendor 时未初始化，如需使用请自行 clone 该 commit。

## 论文 LaTeX 源码

- `Gated Delta Networks Improving Mamba2 with Delta Rule/` — GDN（ICLR 2025）
- `Gated DeltaNet-2 Decoupling Erase and Write in Linear Attention/` — GDN-2

## 改动纪律

对第三方代码的任何改动按 CLAUDE.md 规则执行：以补丁方式注入，在对应目录下建 `patches/` 并用 `PATCHES.md` 记录每一处改动。
