# LSA架构代码 + 训练代码开发计划

日期：2026-08-30
目标：模仿GDN/GDN2的lit_gpt代码，完成LSA及基线的模型架构代码与训练代码，本地单测通过，服务器小规模冒烟。

## 认知对齐（已确认）

- **LSA形态是GQA**：`num_heads`个q头（如16），`num_groups=G`个组（如4）。k、擦除门b、遗忘门g、潜value c、写门w均为**组级**；每组只维护一个潜状态 `T_g ∈ R^{d_k × d_c}`，全层共G份状态。
- **读取**：`o_{t,g,i} = q_{t,g,i} T_g P_{g,i}^T`，P是静态参数 `[H, d_v, d_c]`（按组分块），与token无关。
- **潜维度** `d_c = head_v_dim`（可配），状态容量与GQA基线严格对齐。
- **训练kernel策略**（三选一已对齐，选策略2）：
  - 策略1（入口展开 v=Pc，16个真实state）：仅用于GPU单测交叉验证——与策略2输出在容差内一致，即在真实kernel上验证P可提出性。
  - 策略2（出口乘P，选定）：组级张量(k,g,b,c,w)repeat到H份进`chunk_gdn2`，q每头独立，kernel输出 `[B,T,H,d_c]` 后 `einsum('bthc,hvc->bthv', o, P)` 还原。数学严格等价（docs/research.md归纳证明），训练成本≈MHA（与策略1打平），但数据流与最终GQA化kernel（策略3）完全一致，后续换kernel零重构。
  - 策略3（GQA化新kernel，G份持久state）拆两半：3a decode kernel（`fused_recurrent_gdn2` GQA化，~500行，难度中等）是效率分析实验（推理速度/显存）的硬依赖，排入效率实验前的里程碑；3b chunk训练kernel（~2200行+backward，高风险）默认不做，训练FLOPS对比用解析计算。
  - 注意：策略2下用现有recurrent kernel做质量评估（lm-eval/RULER/JRT）结果正确但缓存16份重复潜state，不省显存；显存/速度数字等3a。
- **代码组织**：`model/lit_gpt/` 自建包（从GDN2的lit_gpt拷贝骨架并改造），Triton kernel通过path shim直接import third_party，不改动、不复制第三方代码。
- **验收**：架构代码 + 训练代码 + 本地CPU单测 + 服务器几百step冒烟。正式训练另起任务。

## 需支持的配置矩阵（本期实现，实验后启动）

主实验在GDN/KDA/GDN2三个骨架上对比（G=4）：
1. MHA baseline（num_groups = num_heads）
2. GQA baseline（num_groups = 4，组级k/v，q每头读组状态）
3. GQA + 增加v_head
4. GQA + expand v_dim（expand_v>1）
5. GQA + LSA（本研究）

统一mixer层参数：`num_heads / num_groups / expand_v / num_v_heads / use_lsa / head_dim / lsa_latent_dim`。

## 工作项

### Phase 1 — model/lit_gpt骨架 + LSA层（核心）

1. `model/lit_gpt/`：从`third_party/GatedDeltaNet-2/lit_gpt`拷贝并适配：
   `config.py`（加mixer选择与LSA/GQA字段）、`model.py`（Block按config选mixer）、
   `rmsnorm.py`、`fused_cross_entropy.py`、`packed_dataset.py`、`speed_monitor.py`、
   `utils.py`、`tokenizer.py`、`rotary.py`（RNN层不用RoPE，attention基线保留）。
2. `model/lit_gpt/kernels.py`（shim）：把`third_party/GatedDeltaNet-2/lit_gpt/gdn2_ops`、
   `third_party/GatedDeltaNet/lit_gpt/gated_delta_rule_ops`加入path并re-export
   `chunk_gdn2 / chunk_kda / chunk_gated_delta_rule`。
3. `model/lit_gpt/mixers/gdn2.py`：改造GatedDeltaNet2层，支持GQA（组级k/v/门，q多头）
   与LSA开关（组级潜c + P参数 + einsum还原）；GVA/expand_v沿用原有能力。
   短conv、o_norm、mamba_init等与GDN2原实现严格对齐。
4. `model/lit_gpt/mixers/naive.py`：纯PyTorch逐token递归参考实现（fp32），
   本地CPU可跑，用于等价性单测（mac无GPU，Triton kernel只能在服务器跑）。
5. GDN、KDA骨架的mixer（`mixers/gdn.py`、`mixers/kda.py`）：同样的GQA/LSA改造。
   优先级次于GDN2（推导在GDN2下），GDN2路径验证通过后铺开。

### Phase 2 — 训练代码

6. `scripts/pretrain.py`：以GDN2的pretrain.py为底（Fabric+FSDP+cosine LR+grad clip），
   数据侧改成读`scripts/data/prepare_fineweb.py`产出的litdata格式（litdata StreamingDataset，
   替换原PackedDataset/内部流式tokenize路径），tokenizer用Llama。
7. `model/lit_gpt/config.py`加约~340M与~1.3B配置：
   每骨架 × {mha, gqa, gqa_vheads, gqa_expandv, gqa_lsa}，参数量对齐（LSA多出的P参数
   与基线的v_proj缩减对冲，冒烟时打印param count核对）。
8. `scripts/train/`：训练启动bash（本地debug单卡 + 服务器slurm模板，对齐
   run-remote-experiment skill的用法）。

### Phase 3 — 测试

9. `tests/test_lsa_equivalence.py`（本地CPU）：
   - naive-LSA（T递归+P还原） vs naive-GVA展开（S_{g,i}=T_g P_{g,i}^T逐头递归）数值一致；
   - P为分块单位阵时LSA退化为GQA baseline；
   - 状态形状/参数量断言。
10. `tests/test_kernel_parity.py`（服务器GPU）：
    - 策略2（出口乘P） vs naive递归，bf16容差内一致；
    - 策略1（入口展开v=Pc） vs 策略2交叉验证，在真实kernel上确认P可提出性；
    - GQA repeat路径同测。
11. `tests/test_model_forward.py`：整模型前向/反向、loss有限性、各配置实例化。

### Phase 4 — 服务器冒烟

12. 同步代码到服务器工作区，跑GPU单测（Phase 3的kernel parity）。
13. ~340M的gdn2_gqa与gdn2_lsa各跑几百step（FineWeb小切片，单卡或单机），
    确认loss正常下降、吞吐/显存记录、ckpt保存恢复可用。

## 并行拆解

- Phase 1的骨架拷贝适配(1,2) 与 naive参考实现(4) 可并行（subagent）。
- Phase 2的数据管线(6) 与 Phase 1的mixer开发可并行。
- GDN/KDA mixer(5) 在GDN2验证后由subagent并行铺开。

## 待定问题（实验启动前再定，不阻塞本期）

- ~~"GQA+增加v_head"的确切语义~~ 已实现（2026-08-30）：k/遗忘/擦除门组级G份、v/w头数=num_v_heads
  （状态数=v头数，与LSA的G份潜状态正对照），待用户核对语义。
- ~~LSA在GDN（scalar beta）骨架下的门形态映射~~ 已解决：GDN/KDA无w门，β标量必须组级，
  P提出性严格成立（GPU parity 7/7验证，含两骨架的策略1v2等价）。
- **expand_v的q头数口径（待用户拍板）**：q=G（读出侧不膨胀，参数恒定，状态容量=MHA）
  vs q=16（读出侧 H×(H/G)dv 膨胀，G=1时o_proj/g_proj爆至1.14B non-emb）。
- ~~100B正式训练超参~~ 已对齐（2026-08-30）：LR 4e-4、warmup=1%、wd 0.1、betas(0.9,0.95)、
  clip 1.0、cosine→LR/10，与GDN/GDN2原版一致；训练量暂定FineWeb-10B（用户依时长预估终定）。
