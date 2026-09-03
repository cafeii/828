# QGDN: A Query-Guided Recall Rule for Gated Delta Networks

> 当前正式结果见 [EXPERIMENTS.md](EXPERIMENTS.md)，训练实现与脚本入口见仓库根目录
> [README.md](../../README.md)。

## 一句话思想

Gated Delta Networks 会对旧状态施加与当前访问需求无关的基础衰减。QGDN 让当前查询 $q_t$ 在 Delta 编辑之前，先把这次衰减刚刚损失的相关读出恢复一部分。

这一步称为 **Recall Rule**。

---

## 1. 问题：全局衰减会无差别削弱当前需要的记忆

设记忆状态为

$$
S_{t-1}\in\mathbb{R}^{d_k\times d_v},
$$

当前归一化查询为

$$
q_t\in\mathbb{R}^{d_k},
\qquad
\lVert q_t\rVert_2=1.
$$

查询从旧状态中读出的内容是

$$
y_t^*=S_{t-1}^{\top}q_t.
$$

标准 GDN 先做标量衰减：

$$
\widetilde S_t=\alpha_tS_{t-1},
\qquad
0\leq\alpha_t\leq1.
$$

衰减后的读出变为

$$
\widetilde y_t
=
\widetilde S_t^{\top}q_t
=
\alpha_tS_{t-1}^{\top}q_t.
$$

因此当前查询刚刚损失的内容是

$$
y_t^*-\widetilde y_t
=
(1-\alpha_t)S_{t-1}^{\top}q_t.
$$

虽然 $\alpha_t$ 已经全局作用于状态，但我们仍然可以在衰减之后，把当前查询对应的读出损失写回去。这就是“保护”的准确含义。

---

## 2. Recall Rule

定义

$$
\boxed{
S_t^{\mathrm{rec}}
=
\widetilde S_t
+
\gamma_tq_t
\left(y_t^*-\widetilde y_t\right)^{\top}
}.
$$

其中 $\gamma_t\in[0,1]$ 是 Recall gate，推荐设为每个 token、每个 head 的标量。

这个公式可以理解为：

- $q_t$ 指定把内容写回哪个地址方向；
- $y_t^*-\widetilde y_t$ 是基础衰减造成的读出误差；
- $\gamma_t$ 控制恢复多少。

它与 Delta Rule 都是秩一误差校正，但目标不同：

- Delta Rule 写入新目标；
- Recall Rule 恢复衰减前的旧读出。

将标量衰减代入后，得到

$$
\begin{aligned}
S_t^{\mathrm{rec}}
&=
\alpha_tS_{t-1}
+
\gamma_t(1-\alpha_t)
q_tq_t^{\top}S_{t-1}
\\
&=
D_t^{(q)}S_{t-1},
\end{aligned}
$$

其中

$$
\boxed{
D_t^{(q)}
=
\alpha_tI
+
\gamma_t(1-\alpha_t)q_tq_t^{\top}
}.
$$

这就是 QGDN 的核心保持算子。

---

## 3. 为什么它能保护当前查询访问的记忆

令

$$
P_t=q_tq_t^{\top}.
$$

因为 $q_t$ 已归一化，所以 $P_t$ 是查询方向 $\operatorname{span}(q_t)$ 上的正交投影。

对任意地址向量 $x$，有

$$
x=P_tx+(I-P_t)x.
$$

QGDN 对这两个分量施加不同保持率：

$$
D_t^{(q)}x
=
\rho_t^{\parallel}P_tx
+
\alpha_t(I-P_t)x,
$$

其中

$$
\rho_t^{\parallel}
=
\alpha_t+
\gamma_t(1-\alpha_t)
=
1-(1-\alpha_t)(1-\gamma_t).
$$

因此：

- 查询正交方向仍按 $\alpha_t$ 衰减；
- 查询平行方向的保持率提升为 $\rho_t^{\parallel}$。

更直接地，Recall 之后的查询读出满足精确恒等式

$$
\boxed{
q_t^{\top}S_t^{\mathrm{rec}}
=
(1-\gamma_t)
q_t^{\top}(\alpha_tS_{t-1})
+
\gamma_tq_t^{\top}S_{t-1}
}.
$$

所以：

- $\gamma_t=0$：不恢复，退化为原始 GDN；
- $0<\gamma_t<1$：部分恢复；
- $\gamma_t=1$：Recall 步之后完全恢复衰减前的当前查询读出。

这里保护的是“当前查询可读出的子空间”，不是状态矩阵的某些固定行。

---

## 4. 为什么使用 $q_tq_t^{\top}$，而不是 $\operatorname{Diag}(q_t)$

$q_tq_t^{\top}$ 表示查询方向本身，而 $\operatorname{Diag}(q_t)$ 只是对预先选定的坐标轴分别缩放。

直接使用

$$
\operatorname{Diag}(q_t)
$$

并不合适，因为 $q_{t,i}$ 可以为负，可能产生带符号的“保持系数”。

改成

$$
\operatorname{Diag}(q_t^2)
$$

虽然非负，但仍然丢失所有跨坐标项 $q_{t,i}q_{t,j}$。两者的差异满足

$$
\left\lVert
q_tq_t^{\top}-\operatorname{Diag}(q_t^2)
\right\rVert_F^2
=
1-\sum_iq_{t,i}^4.
$$

只有查询接近 one-hot 时，它们才近似相同。

此外，投影形式在地址空间旋转下自然变换：

$$
(Uq_t)(Uq_t)^{\top}
=
U(q_tq_t^{\top})U^{\top}.
$$

对角门一般不具备这种性质。因此主方法应使用 $q_tq_t^{\top}$；$\operatorname{Diag}(q_t^2)$ 可以作为坐标门控消融，但不是 Recall Rule 的等价替代。

---

## 5. 完整的 QGDN 更新

Recall Rule 应放在原始 Delta 编辑之前：

$$
S_t^{\mathrm{rec}}=D_t^{(q)}S_{t-1},
$$

$$
S_t
=
S_t^{\mathrm{rec}}
+
\beta_tk_t
\left(
v_t-(S_t^{\mathrm{rec}})^{\top}k_t
\right)^{\top}.
$$

合并后为

$$
\boxed{
S_t
=
\left(I-\beta_tk_tk_t^{\top}\right)
D_t^{(q)}S_{t-1}
+
\beta_tk_tv_t^{\top}
}.
$$

这个顺序的语义很清楚：

1. 基础衰减；
2. Recall 当前查询刚刚损失的旧读出；
3. 用原始 Delta Rule 编辑并写入新内容。

需要特别区分两个结论：

- Recall Rule 可以保护旧读出免受基础衰减；
- 随后的 Delta 编辑仍可能改变该读出。

所以即使 $\gamma_t=1$，也不能声称完整一步更新后 $q_t^{\top}S_t$ 必然保持不变。

---

## 6. 稳定性

$D_t^{(q)}$ 的两个特征值为

$$
\rho_t^{\parallel}
=
1-(1-\alpha_t)(1-\gamma_t)
$$

和

$$
\rho_t^{\perp}=\alpha_t.
$$

当

$$
0\leq\alpha_t,\gamma_t\leq1
$$

时，

$$
\left\lVert D_t^{(q)}\right\rVert_2
=
\rho_t^{\parallel}
\leq1.
$$

若 $\lVert k_t\rVert_2=1$ 且 $0\leq\beta_t\leq1$，则完整齐次传播矩阵

$$
A_t
=
\left(I-\beta_tk_tk_t^{\top}\right)D_t^{(q)}
$$

满足

$$
\lVert A_t\rVert_2
\leq
\rho_t^{\parallel}
\leq1.
$$

因此标量 QGDN 不会因为状态传播本身产生指数爆炸。

严格收缩裕度为

$$
1-\rho_t^{\parallel}
=
(1-\alpha_t)(1-\gamma_t).
$$

这也暴露了主要风险：如果 $\gamma_t$ 长期接近 $1$，模型虽然记得更多，但有效遗忘裕度会接近零。此时有界写入仍可能长期累积。

所以应区分：

- **非扩张**：排除齐次系统的指数爆炸；
- **严格收缩**：进一步给出持续有界写入下的状态上界。

---

## 7. 为什么主版本优先采用标量或逐头衰减

一次 KV 外积写入是

$$
k_\tau v_\tau^{\top}.
$$

如果后续采用标量衰减，它在时刻 $t$ 的贡献为

$$
\left(
\prod_{s=\tau+1}^t\alpha_s
\right)
k_\tau v_\tau^{\top}.
$$

历史 key 的方向不变，只是整体变弱。

如果采用 GDN-2 式逐行对角衰减，则贡献为

$$
\left(
\prod_{s=\tau+1}^tD_s
\right)
k_\tau v_\tau^{\top}.
$$

此时历史 key 等价地变为

$$
k_{\tau\rightarrow t}
=
\left(
\prod_{s=\tau+1}^tD_s
\right)k_\tau,
$$

它通常不再与原来的 $k_\tau$ 平行。也就是说，逐行衰减会持续扭曲历史地址，而不仅仅是遗忘内容。

因此更稳妥的研究定位是：

- 标量或逐头 GDN + Recall Rule：QGDN 主方法；
- 逐行 GDN-2 + Recall Rule：表达力更强但语义更复杂的扩展；
- 是否需要逐行衰减，应由消融实验回答。

---

## 8. GDN-2 的稳定推广

对一般对称保持矩阵

$$
0\preceq D_t\preceq I,
$$

定义

$$
W_t=I-D_t,
$$

$$
r_t=W_tq_t,
$$

$$
\delta_t=q_t^{\top}W_tq_t.
$$

当 $\delta_t>0$ 时，稳定推广为

$$
\boxed{
D_t^{(q)}
=
D_t
+
\gamma_t
\frac{r_tr_t^{\top}}{\delta_t}
}.
$$

当 $\delta_t=0$ 时，令

$$
D_t^{(q)}=D_t.
$$

它满足精确读出插值：

$$
\boxed{
q_t^{\top}D_t^{(q)}S
=
(1-\gamma_t)q_t^{\top}D_tS
+
\gamma_tq_t^{\top}S
}.
$$

同时满足

$$
\boxed{
0\preceq D_t
\preceq D_t^{(q)}
\preceq I
}.
$$

所以

$$
\left\lVert D_t^{(q)}\right\rVert_2\leq1.
$$

若 $D_t=\alpha_tI$，该公式恰好退化为主公式

$$
D_t^{(q)}
=
\alpha_tI
+
\gamma_t(1-\alpha_t)q_tq_t^{\top}.
$$

一个更简单的非对称形式

$$
D_t+
\gamma_tq_tq_t^{\top}(I-D_t)
$$

虽然也满足读出插值，但对一般对角 $D_t$ 不保证谱范数小于等于 $1$，因此只应作为消融项。

---

## 9. 实现形式

对一般稳定推广，令

$$
u_t=\frac{r_t}{\sqrt{\delta_t}},
$$

则

$$
D_t^{(q)}=D_t+\gamma_tu_tu_t^{\top}.
$$

因此不需要显式构造 $d_k\times d_k$ 的稠密矩阵：

$$
D_t^{(q)}S
=
D_tS
+
\gamma_tu_t(u_t^{\top}S).
$$

额外操作仍然是秩一更新。

数值上，当

$$
\delta_t=q_t^{\top}(I-D_t)q_t
$$

接近零时，应使用显式分支跳过 Recall 校正。若简单把分母替换成 $\delta_t+\varepsilon$，实现可以更稳，但将不再严格满足精确读出插值恒等式。

---

## 10. 推荐定义与实验顺序

主文首先定义：

$$
\boxed{
D_t^{(q)}
=
\alpha_tI
+
\gamma_t(1-\alpha_t)q_tq_t^{\top}
}.
$$

并使用：

$$
\boxed{
S_t
=
\left(I-\beta_tk_tk_t^{\top}\right)
D_t^{(q)}S_{t-1}
+
\beta_tk_tv_t^{\top}
}.
$$

实验建议依次验证：

1. 标量 GDN 与标量 QGDN；
2. $q_tq_t^{\top}$ 与 $\operatorname{Diag}(q_t^2)$；
3. 固定、逐头和逐 token 的 $\gamma_t$；
4. Recall 前置与后置；
5. GDN-2 的稳定对称推广与非对称消融；
6. 长序列状态范数、梯度范数和有效记忆寿命。

最终标题为：

> **QGDN: A Query-Guided Recall Rule for Gated Delta Networks**

最终的一句话表述为：

> QGDN uses the current query to recall the portion of its old readout that global decay would otherwise erase, before the standard Delta update edits the memory.
