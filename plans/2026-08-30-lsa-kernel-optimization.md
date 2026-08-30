# LSA专用kernel优化计划（策略3，后续任务）

日期：2026-08-30
状态：**后置**——当前训练/质量评估全部走策略2（出口乘P复用`chunk_gdn2`），本计划在效率分析实验前启动。
前置依赖：策略2路径训练与GPU parity单测通过（见 2026-08-30-lsa-litgpt-dev.md）。

## 目标

实现GQA化的LSA kernel：潜状态只持久化G份 `T_g ∈ R^{d_k×d_c}`，与q、P计算时瞬时扩展到H头，
兑现LSA的推理显存/速度收益。

## 拆解

### 3a：decode kernel（硬依赖，优先）

改造 `fused_recurrent_gdn2`（~500行）为GQA形态：

- state张量 `[N, G, d_k, d_c]`（原为 `[N, H, d_k, d_v]`），更新（擦除/遗忘/写入）按组算一次；
- q读取per-head：`o_{g,i} = q_{g,i} T_g`，输出 `[B, T, H, d_c]`，×P仍在kernel外（与策略2数据流一致）；
- 组内头的q读取可在同一program内循环，T_g载入一次复用I次；
- 仅推理（无backward），风险低。

验收：与策略2路径（repeat进现有kernel）输出bf16容差一致；state显存降I倍；
接入lm-eval/RULER/JRT推理路径与效率分析的decode吞吐测试。

### 3b：chunk训练kernel（默认不做）

`chunk_gdn2` GQA化（~2200行，WY变换+sub-chunk+autotune+backward）。
反向需注意：dk/dg/db/d state在组内跨头求和，dq/do保持per-head。
风险高、周期以周计。触发条件：1.3B正式训练成本压力显著，且3a已验证数值方案。
训练FLOPS对比实验用解析计算，不依赖此项。

## 效率分析实验对接

- 推理速度/显存：3a落地后，对比 MHA / GQA / GQA+LSA 的decode吞吐与state显存占用；
- 训练FLOPS：解析计算（策略2实测值代表MHA量级上界，3b未做时注明）。
