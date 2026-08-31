# GDN / QGDN 约 340M 对照实验

目标是检验 Query-Guided Recall 是否提高效果，不预设结果为正。主方法严格采用
`research/qgdn/DERIVATION_PACKAGE.md` 的标量 GDN 推导，不包含 GDN2、GQA 或 LSA。

## 模型与机制

两者共同使用 20 层、hidden=1024、16 个独立头、key/value head_dim=64、SwiGLU
intermediate=2816、ShortConv=4、32k 词表、非共享输入 embedding / lm_head。
包含两套词表矩阵的参数量：GDN **344,353,984**，QGDN **344,681,984**；
额外 **328,000** 个参数（约 0.095%）来自每层 hidden→head 的 gamma 投影及 bias。
同 seed 的所有公共参数初始化完全相同，训练记录其 SHA256。
旧 `gdn_mha_340M` 名称保留，实际约400M，**不要用于本组主对照**。

主更新：

```
S_rec = alpha * S + gamma * (1-alpha) * q * (q^T S)
S_new = S_rec + beta * k * (v - k^T S_rec)
output = q^T S_new / sqrt(head_dim)
```

q/k 使用 L2 归一化。gamma 是每 token / head 的 sigmoid 标量，投影权重初始化为0，
bias 初始化为 logit(0.1)。beta∈[0,1]；不允许 `allow_neg_eigval`。
主实现复用 FLA DPLR chunk：每个真实 token 先 Recall、再 Delta，共 **2T 个虚拟步骤**，
只读第二步输出。状态大小不变，无 alpha 求逆、无显式 K×K 保持矩阵。
这是可训练的初始实现，不承诺吞吐接近优化后的原始 GDN；必须报告实测成本。
`fused_recurrent` 只用于无梯度推理；训练用 chunk。参考实现用于小规模数值验证。
目前不支持 QGDN 的 GQA/LSA、GDN2 扩展。GPT 层面的增量生成缓存沿用旧代码的限制，
评测入口始终完整前向，不宣称已经支持增量生成。

## 对照矩阵

| variant | config | 控制的问题 |
|---|---|---|
| gdn | gdn_control_340M | 原始标准 GDN |
| qgdn | qgdn_340M | 当前 query 方向上的 Recall，主方法 |
| key | qgdn_key_340M | 相同门参数、相同两步后端，改用 key 方向 |
| isotropic | qgdn_isotropic_340M | 相同门参数，将所有方向保持率都提高为 alpha+gamma(1-alpha) |
| fixed | qgdn_fixed_340M | gamma=0.1，无新增参数，分离 Recall 与动态门 |
| head | qgdn_head_340M | 每层每头一个可学习 gamma，共新增320参数 |
| zero | qgdn_zero_340M | gamma=0，检验两步后端退化为 GDN；主要用于实现校验 |

至少先跑 gdn/qgdn；效果有信号后补 key、isotropic、fixed。
isotropic 不保持秩一修正量的迹与 QGDN 相同：它是“全局少遗忘”对照，不是所有意义上的等量干预。
单独训练各变体；不要把测试时突然关闭 Recall 的性能下降当作充分的因果证据。

## 预注册训练设置

主任务：同一语料、同一 tokenizer 的自回归语言模型，从头训练；context=4096，
global batch=128 sequences，默认 micro batch=1（可在所有变体上统一调整）。
AdamW lr=4e-4、betas=(0.9,0.95)、weight decay=0.1、clip=1；norm/bias、A_log/dt_bias
及声明 no_weight_decay 的参数不做 weight decay。BF16 mixed、DDP、逐 block activation checkpointing。

- smoke：tiny 模型，MQAR，3 steps，仅测试链路。
- pilot：512 steps，**268,435,456** 个预测 token，先 seed3407。用于检查稳定性和成本。
- main：19,073 steps，**9,999,745,024** 个预测 token，seeds **3407 / 42 / 2026**。

每步 tokens=global batch×context=524,288。不会在不完整梯度累积处停止。
warmup=floor(steps×1%)，至少1步；随后 cosine 到初始 lr 的10%。每个变体使用同样完整日程。
pilot/main 是预设预算，**生成计划不启动训练，预算可在主实验前统一修改**。
主结论比较固定预算的 final checkpoint，而非挑选各自最优 seed 或最佳中间 checkpoint。
调参只看开发验证集、给予各变体相同搜索预算，最终主实验设置与结果应完整保留。

## 数据准备

现有继承的 litdata 数据只有 train，不能直接声称存在独立验证集。本实验使用独立的
uint16 token 文件和 manifest，以显式全局样本索引确保跨累积、跨 rank 和恢复后顺序一致。
这是一套新数据接口，**不会原地修改或自动转换同事的 litdata 数据**。

在 CPU Slurm 作业内运行，所有参数必须指向实际资源；脚本不下载数据、不提交作业：

```bash
python scripts/qgdn/prepare_data.py \
  --input-dir /absolute/path/to/fineweb/parquet \
  --tokenizer /absolute/path/to/llama2-32k-tokenizer \
  --output-dir /absolute/path/to/new/qgdn-data \
  --val-files 1
```

按排序后 parquet 文件的尾部划出 validation，训练和验证分别 token 化；每篇文档追加 EOS、无 BOS。
记录源文件、token 文件、tokenizer 文件的哈希；拒绝同一源文件内容出现在两个划分中。
该检查不等于跨文件的文档去重，正式研究仍需确认语料本身的去重与 benchmark 污染情况。
训练按4097 token 切块，用前4096预测后4096；尾部不完整块丢弃。
训练样本顺序由 dataset hash、seed、全局序号确定，耗尽后按下个 epoch 的固定排列继续，记录总预算。
验证固定使用前2560个完整块（约10.49M个预测token）；数据较少时明确记录实际数量，不循环重复。
rank0训练前验证完整 token 文件哈希，所有 rank 核对大小与划分。

## 创建计划与启动方式

开发仓库在 QGDN 分支；提交自己的改动后，用个人 `run-remote-experiment` Skill 固定快照。
下列命令在实验快照中运行；路径必须替换成真实绝对路径，输出放入该实验的 outputs。

```bash
python scripts/qgdn/matrix.py --stage pilot --task lm \
  --variants gdn qgdn --devices 8 \
  --data-manifest /absolute/path/to/qgdn-data/manifest.json \
  --out-root /absolute/path/to/experiment/outputs/runs \
  --plan-dir /absolute/path/to/experiment/outputs/plan
```

产出 `plan.json` 与 `commands.txt`，**仅生成命令，不提交和执行**。
每条命令通过个人 Skill 生成独立 Slurm 作业，遵守节点 best-fit、姓名/组名、时长与回收规则。
多卡仅支持单节点 DDP；未实现 FSDP 或跨节点启动。不要调用原有同事路径的训练模板。
main 改为 `--stage main`；消融例如 `--variants gdn qgdn key isotropic fixed`。

训练产物：`run.json`、`metrics.jsonl`、可恢复的 `checkpoint.pt`、`summary.json`、
最终 `model_final.pt`。保存为临时文件后原子替换；旧 checkpoint 按约定只保留 latest，
需要保留中间检查点时应在正式实验前扩展保存策略。日志包含 loss、实际tokens、grad norm、
step耗时、吞吐、显存以及 gamma、alpha、遗忘余量诊断（门统计取记录步最后一个micro batch）。
吞吐同时受后端、checkpointing、编译和硬件影响；不要把 walltime 全部解释为算法计算量。

恢复必须显式传 `--resume /absolute/path/to/checkpoint.pt`，继续使用原 output 目录，
模型、代码提交、数据、seed、拓扑、完整预算与超参必须一致。任意加载失败立即退出，不自动从头重训。
`--stop-after-step N` 可模拟中断并保存；它不改变完整预算或 LR 日程。
旧日志中未进入 checkpoint 的步骤可能在恢复后重复，分析曲线时按 step 取最后一次记录；
主汇总使用 final summary，不会把 paused 结果当作 completed。

## 效果评测与判读

主指标是独立验证集 final NLL / perplexity。报告三组配对种子的全部差值、均值和样本标准差，
不能凭一个 seed 或一次 smoke 证明提升。三种子仍是有限证据；小差异需更多种子或更大评测集。

```bash
python scripts/qgdn/summarize.py /absolute/path/to/gdn-run /absolute/path/to/qgdn-run \
  --output /absolute/path/to/paired-results.json

python scripts/qgdn/evaluate.py --checkpoint /absolute/path/to/run/model_final.pt \
  --data-manifest /absolute/path/to/qgdn-data/manifest.json \
  --lengths 1024 4096 8192 --output /absolute/path/to/context-evaluation.json
```

汇总拒绝不完整预算、不匹配的代码/数据/超参/公共初始化、重复 seed 权重及不一致验证token数。
长上下文评测记录各长度及位置四分位的 NLL；不同长度的 chunk 边界不同，应在同长度上做模型间配对。

辅助机制任务使用 `train.py --task mqar`：随机 key/value 写入，间隔 filler 后多次查询；
只在答案位置计算 loss，验证使用与训练独立的固定seed。`--mqar-overwrite` 加入对旧key的重复写入，
正确答案取最后一次写入。`evaluate.py --lengths ...` 可增加读取距离。
任务与真实语料预训练分开，需对两模型使用同一预算；合成任务收益不能代替语言建模收益。
不要直接给未在该符号任务上训练的语言模型评分并声称是检索能力。

有效性的证据应同时包含：主语言建模指标可重复改善、query 相对 key/全局减弱遗忘对照的表现、
固定/动态 gamma 消融、长距离诊断、覆盖旧记忆是否退化，以及实际训练时间/显存。
若无收益或成本过高，照实报告并据此修订假设，不能通过只保留有利实验来“证明”机制。

## 验证入口与当前边界

```bash
python scripts/qgdn/validate.py --output /absolute/path/to/experiment/outputs/validation --full-model
```

该入口必须在已分配的短 Slurm 作业内运行：稠密公式/梯度、GDN退化、读出插值、状态携带、
BF16 chunk 输出/状态/梯度、数据顺序与污染防护、tiny LM/MQAR训练、严格恢复、长距离评测，
两卡可见时验证DDP；`--full-model` 额外对两个约340M模型各跑一个4096长度的优化器步。
测试通过只表示实现/训练链路通过指定检查，**不表示 QGDN 已获得效果提升**。
实际验证 commit、作业号与结果另见回收的 validation.json / 日志，不能仅凭测试源码宣称全部已通过。
