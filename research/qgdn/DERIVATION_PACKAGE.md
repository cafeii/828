# QGDN: A Query-Guided Recall Rule for Gated Delta Networks

## 1. Target

本文推导一种用于 Gated Delta Networks（GDN）的查询引导记忆保持机制。核心目标不是让查询直接控制写入，而是让当前查询参与决定：**基础衰减之后，哪些旧记忆应当被立即召回并恢复。**

我们将这一机制称为 **Recall Rule**，将完整模型称为：

> **QGDN: A Query-Guided Recall Rule for Gated Delta Networks**

QGDN 展开为 **Query-Guided DeltaNet**。

本文区分三种层级：

1. **主方法**：标量或逐头基础衰减上的 Recall Rule；
2. **稳定推广**：适用于一般对称衰减矩阵、尤其是 GDN-2 对角衰减的 Recall Rule；
3. **消融项**：形式更简单、但一般不保证收缩性的非对称查询保护公式。

---

## 2. Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.**

结论如下：

- 标量 GDN 上的 Recall Rule 可以从“恢复基础衰减造成的当前查询读出损失”严格推出；
- 其矩阵形式是一个沿查询方向的秩一校正，并且在常见门值范围内保持非扩张性；
- 对一般对角衰减，直接使用旧的非对称公式可能破坏谱范数稳定性；
- 经过对称归一化后，可得到同时满足精确读出插值与收缩性保证的稳定推广；
- QGDN 只能保护旧记忆免受**基础衰减**，不能自动抵消随后 Delta 编辑带来的覆盖。

---

## 3. Invariant Object

### 3.1 语义不变量

QGDN 关心的不是某一行状态是否保持不变，而是当前查询能够读出的内容是否因基础衰减而丢失。

设记忆状态为

$$
S_{t-1}\in\mathbb{R}^{d_k\times d_v},
$$

归一化查询为

$$
q_t\in\mathbb{R}^{d_k},
\qquad
\lVert q_t\rVert_2=1.
$$

衰减前的查询读出为

$$
y_t^*=S_{t-1}^{\top}q_t.
$$

若基础保持算子为 $D_t$，衰减后的状态与读出分别为

$$
\widetilde S_t=D_tS_{t-1},
$$

$$
\widetilde y_t
=
\widetilde S_t^{\top}q_t
=
S_{t-1}^{\top}D_tq_t,
$$

其中假设 $D_t$ 为对称矩阵。基础衰减对当前读出造成的损失为

$$
e_t
=
y_t^*-\widetilde y_t
=
S_{t-1}^{\top}(I-D_t)q_t.
$$

Recall Rule 的目标就是把 $e_t$ 的一部分写回状态。

### 3.2 动力学不变量

完整状态更新写成仿射系统：

$$
S_t=A_tS_{t-1}+B_t.
$$

因此稳定性分析必须分别检查：

- 齐次传播矩阵 $A_t$ 是否会放大状态与状态梯度；
- 写入项 $B_t$ 是否会在非严格收缩系统中长期累积。

---

## 4. Assumptions

除非另有说明，本文采用以下假设：

1. 查询已经归一化：

   $$
   \lVert q_t\rVert_2=1.
   $$

2. 标量 GDN 的基础保持算子为

   $$
   D_t=\alpha_t I,
   \qquad
   0\leq\alpha_t\leq1.
   $$

3. GDN-2 的基础保持算子为

   $$
   D_t=\operatorname{Diag}(d_t),
   \qquad
   0\leq d_{t,i}\leq1.
   $$

4. Recall gate 为标量：

   $$
   0\leq\gamma_t\leq1.
   $$

5. 标准 GDN 的 Delta 步使用归一化 key 与

   $$
   0\leq\beta_t\leq1.
   $$

6. 对一般矩阵的稳定推广要求

   $$
   0\preceq D_t\preceq I,
   \qquad
   D_t=D_t^{\top}.
   $$

---

## 5. Notation

| 符号 | 含义 |
|---|---|
| $S_t$ | 时刻 $t$ 的记忆矩阵 |
| $q_t$ | 当前归一化查询 |
| $k_t,v_t$ | 当前写入的 key 与 value |
| $D_t$ | 原始基础保持算子 |
| $D_t^{(q)}$ | 经过 Recall Rule 修正后的查询引导保持算子 |
| $\alpha_t$ | 标量 GDN 的基础保持率 |
| $\gamma_t$ | Recall 强度，默认是每个 token、每个 head 的标量 |
| $P_t=q_tq_t^{\top}$ | 当前查询方向上的正交投影矩阵 |
| $W_t=I-D_t$ | 基础遗忘算子 |
| $r_t=W_tq_t$ | 当前查询实际受到遗忘的方向 |
| $\delta_t=q_t^{\top}W_tq_t$ | 当前查询上的总遗忘量 |

---

## 6. Derivation Strategy

推导遵循以下顺序：

1. 从当前查询读出在基础衰减前后的差值出发；
2. 用一个秩一记忆更新恢复该差值的一部分；
3. 在标量衰减下化简为闭式保持矩阵；
4. 证明其查询读出插值性质与谱范数稳定性；
5. 再推广到一般对称衰减矩阵；
6. 最后把 Recall Rule 与原有 Delta 编辑组合起来。

该顺序强调：QGDN 首先是一条**读出恢复规则**，矩阵门只是它的等价实现形式。

---

## 7. Derivation Map

$$
\text{基础衰减}
\;\Longrightarrow\;
\text{当前查询读出损失}
\;\Longrightarrow\;
\text{Recall Rule}
\;\Longrightarrow\;
D_t^{(q)}
\;\Longrightarrow\;
\text{Delta 编辑与新写入}.
$$

对主方法，逻辑链为

$$
\alpha_tS_{t-1}
\;\Longrightarrow\;
(1-\alpha_t)S_{t-1}^{\top}q_t
\;\Longrightarrow\;
\gamma_tq_t
\bigl(y_t^*-\widetilde y_t\bigr)^{\top}
\;\Longrightarrow\;
\left[
\alpha_tI+
\gamma_t(1-\alpha_t)q_tq_t^{\top}
\right]S_{t-1}.
$$

---

## 8. Main Derivation

### 8.1 为什么全局衰减仍然可以被“保护”

在标量 GDN 中，基础衰减是

$$
\widetilde S_t=\alpha_tS_{t-1}.
$$

因此，衰减前后当前查询的读出分别为

$$
y_t^*=S_{t-1}^{\top}q_t,
$$

$$
\widetilde y_t
=
\widetilde S_t^{\top}q_t
=
\alpha_tS_{t-1}^{\top}q_t.
$$

读出损失是

$$
y_t^*-\widetilde y_t
=
(1-\alpha_t)S_{t-1}^{\top}q_t.
$$

所以，保护并不是让 $\alpha_t$ 停止衰减，也不是逐行决定“哪些元素不衰减”。它是在衰减以后，沿当前查询方向将丢失的读出部分恢复回来。

### 8.2 Recall Rule

定义 Recall Rule：

$$
\boxed{
S_t^{\mathrm{rec}}
=
\widetilde S_t
+
\gamma_t q_t
\left(y_t^*-\widetilde y_t\right)^{\top}
}.
$$

这与 Delta Rule 的“误差校正”结构相似，但两者目标不同：

- Delta Rule 用新目标修正 key 对应的内容；
- Recall Rule 用衰减前的旧读出作为目标，恢复当前查询刚刚损失的内容。

代入标量衰减可得

$$
\begin{aligned}
S_t^{\mathrm{rec}}
&=
\alpha_tS_{t-1}
+
\gamma_tq_t
\left[
(1-\alpha_t)S_{t-1}^{\top}q_t
\right]^{\top}
\\
&=
\left[
\alpha_tI
+
\gamma_t(1-\alpha_t)q_tq_t^{\top}
\right]S_{t-1}.
\end{aligned}
$$

因此定义

$$
\boxed{
D_t^{(q)}
=
\alpha_tI
+
\gamma_t(1-\alpha_t)q_tq_t^{\top}
}.
$$

并写作

$$
S_t^{\mathrm{rec}}=D_t^{(q)}S_{t-1}.
$$

这是 QGDN 的核心公式。

### 8.3 精确的查询读出插值

因为 $\lVert q_t\rVert_2=1$，所以

$$
q_t^{\top}D_t^{(q)}
=
\left[
\alpha_t+
\gamma_t(1-\alpha_t)
\right]q_t^{\top}.
$$

令

$$
\rho_t^{\parallel}
=
\alpha_t+
\gamma_t(1-\alpha_t)
=
1-(1-\alpha_t)(1-\gamma_t),
$$

则

$$
q_t^{\top}S_t^{\mathrm{rec}}
=
\rho_t^{\parallel}q_t^{\top}S_{t-1}.
$$

等价地，

$$
q_t^{\top}S_t^{\mathrm{rec}}
=
(1-\gamma_t)
q_t^{\top}(\alpha_tS_{t-1})
+
\gamma_tq_t^{\top}S_{t-1}.
$$

因此：

- $\gamma_t=0$：完全退化为原始 GDN 衰减；
- $0<\gamma_t<1$：恢复部分衰减前读出；
- $\gamma_t=1$：在 Recall 步中精确恢复当前查询的衰减前读出。

这里的“保护”是一个严格的读出恒等式，而不是启发式说法。

### 8.4 为什么使用 $q_tq_t^{\top}$

定义

$$
P_t=q_tq_t^{\top}.
$$

由于查询归一化，$P_t$ 满足

$$
P_t^{\top}=P_t,
\qquad
P_t^2=P_t.
$$

因此它是查询方向 $\operatorname{span}(q_t)$ 上的正交投影。任意地址向量 $x$ 都可分解为

$$
x=P_tx+(I-P_t)x.
$$

QGDN 对两部分施加不同保持率：

$$
D_t^{(q)}x
=
\rho_t^{\parallel}P_tx
+
\alpha_t(I-P_t)x.
$$

也就是说：

- 与当前查询平行的记忆方向，保持率从 $\alpha_t$ 提升到 $\rho_t^{\parallel}$；
- 与当前查询正交的方向，仍按 $\alpha_t$ 衰减。

这正是“查询引导保护”的几何含义。

### 8.5 为什么不使用 $\operatorname{Diag}(q_t)$

直接使用

$$
\operatorname{Diag}(q_t)
$$

有三个问题：

1. $q_{t,i}$ 可能为负，使“保持系数”变成带符号缩放；
2. 它依赖坐标轴，不表示查询张成的一维子空间；
3. 它无法恢复跨坐标项 $q_{t,i}q_{t,j}$ 对读出的共同贡献。

即使改成非负的

$$
\operatorname{Diag}(q_t^2),
$$

也只是在坐标维度上分配保持率。它与真正的查询投影之差为

$$
\left\lVert
q_tq_t^{\top}-\operatorname{Diag}(q_t^2)
\right\rVert_F^2
=
1-\sum_iq_{t,i}^4.
$$

只有当 $q_t$ 接近 one-hot 时，两者才接近。

更强地，若希望某个对角门 $G_t$ 对任意状态都满足完整查询保护

$$
q_t^{\top}G_tS=q_t^{\top}S,
$$

那么必须有

$$
G_{t,ii}=1
\quad
\text{对所有 }q_{t,i}\neq0.
$$

对稠密查询，这几乎等于关闭所有相关坐标上的衰减，失去选择性。

此外，投影形式在地址空间正交旋转 $U$ 下满足

$$
(Uq_t)(Uq_t)^{\top}
=
U(q_tq_t^{\top})U^{\top},
$$

而对角形式通常不满足这种等变性。因此，$q_tq_t^{\top}$ 是主方法中更自然的对象；$\operatorname{Diag}(q_t^2)$ 应作为“坐标时间尺度门控”的独立消融项。

### 8.6 与原始 GDN 更新组合

Recall Rule 先处理基础衰减，原始 Delta Rule 再进行当前 token 的编辑与写入。

对标准 GDN，可写为

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

等价的两步形式为

$$
S_t^{\mathrm{rec}}=D_t^{(q)}S_{t-1},
$$

$$
S_t
=
S_t^{\mathrm{rec}}
+
\beta_tk_t
\left(v_t-(S_t^{\mathrm{rec}})^{\top}k_t\right)^{\top}.
$$

当 $\gamma_t=0$ 时，

$$
D_t^{(q)}=\alpha_tI,
$$

因此 QGDN 精确退化为原始 GDN。

必须注意：$\gamma_t=1$ 只保证 **Recall 步之后** 的当前查询读出不受基础衰减影响。随后的 Delta 编辑仍可能改变该读出，尤其当 $k_t$ 与 $q_t$ 高度相关时。

### 8.7 标量 QGDN 的稳定性

因为 $D_t^{(q)}$ 在查询方向和正交补上的特征值分别为

$$
\rho_t^{\parallel}
=
1-(1-\alpha_t)(1-\gamma_t),
$$

$$
\rho_t^{\perp}=\alpha_t,
$$

所以

$$
0\leq\alpha_t
\leq
\rho_t^{\parallel}
\leq1,
$$

且

$$
\left\lVert D_t^{(q)}\right\rVert_2
=
\rho_t^{\parallel}
\leq1.
$$

若 $\lVert k_t\rVert_2=1$ 且 $0\leq\beta_t\leq1$，则

$$
\left\lVert I-\beta_tk_tk_t^{\top}\right\rVert_2=1.
$$

因此齐次传播矩阵

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

这说明 QGDN 不会仅因齐次状态传播而产生指数爆炸。

严格收缩裕度是

$$
1-\rho_t^{\parallel}
=
(1-\alpha_t)(1-\gamma_t).
$$

如果存在常数 $\varepsilon_\alpha,\varepsilon_\gamma>0$，使得

$$
\alpha_t\leq1-\varepsilon_\alpha,
\qquad
\gamma_t\leq1-\varepsilon_\gamma,
$$

则

$$
\lVert A_t\rVert_2
\leq
1-\varepsilon_\alpha\varepsilon_\gamma
<1.
$$

若写入满足

$$
\lVert B_t\rVert_F\leq\overline b,
$$

并且 $\lVert A_t\rVert_2\leq\overline\rho<1$，则

$$
\lVert S_t\rVert_F
\leq
\overline\rho^t\lVert S_0\rVert_F
+
\frac{\overline b}{1-\overline\rho}.
$$

但若只能证明 $\lVert A_t\rVert_2\leq1$，则有界写入仍可能线性累积。因此：

> 非扩张性可以排除指数爆炸，但不等同于完整的 BIBO 有界性。

当 $\gamma_t=1$ 时，Recall 算子在 $q_t$ 上存在单位保持模态。随后 Delta 编辑对该方向的收缩程度满足

$$
\left\lVert
\left(I-\beta_tk_tk_t^{\top}\right)q_t
\right\rVert_2^2
=
1-
\beta_t(2-\beta_t)
(k_t^{\top}q_t)^2.
$$

若 $q_t\perp k_t$，则当前 Delta 步不会消除这个单位模态。

### 8.8 为什么主方法优先采用标量或逐头衰减

KV 外积写入为

$$
k_\tau v_\tau^{\top}.
$$

若后续使用标量衰减，那么该历史写入在时刻 $t$ 的贡献是

$$
\left(
\prod_{s=\tau+1}^t\alpha_s
\right)
k_\tau v_\tau^{\top}.
$$

它只改变强度，不改变历史 key 的方向。

若使用逐行对角衰减，则贡献变为

$$
\left(
\prod_{s=\tau+1}^tD_s
\right)
k_\tau v_\tau^{\top}.
$$

此时历史 key 等价地变成

$$
k_{\tau\rightarrow t}
=
\left(
\prod_{s=\tau+1}^tD_s
\right)k_\tau,
$$

它通常不再与原始 $k_\tau$ 平行。换言之，逐行衰减不仅让内容变弱，还持续改变历史地址的方向，从而改变检索几何。

因此：

- 标量或逐头衰减具有更清楚的 KV 语义，应作为 QGDN 主版本；
- GDN-2 的逐行衰减更有表达力，但其“按地址通道独立遗忘”的必要性需要实验验证；
- Recall Rule 与 GDN-2 的组合应视为重要扩展，而不是定义 QGDN 所必需的部分。

### 8.9 一般对称衰减下的稳定 Recall Rule

现在考虑

$$
0\preceq D_t\preceq I,
\qquad
D_t=D_t^{\top}.
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

$r_t$ 是查询中实际被基础衰减削弱的方向，$\delta_t$ 是该方向上的总遗忘量。

当 $\delta_t>0$ 时，定义稳定查询引导保持算子

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

等价地，令 $P_t=q_tq_t^{\top}$，则

$$
D_t^{(q)}
=
D_t
+
\gamma_t
\frac{
(I-D_t)P_t(I-D_t)
}{
\operatorname{tr}\left[P_t(I-D_t)\right]
}.
$$

当 $\delta_t=0$ 时，说明当前查询没有受到基础衰减，应定义

$$
D_t^{(q)}=D_t.
$$

#### 8.9.1 从 Recall Rule 推导

一般衰减下的读出误差仍为

$$
e_t
=
S_{t-1}^{\top}r_t.
$$

选择归一化写回地址

$$
h_t=\frac{r_t}{\delta_t},
$$

则

$$
q_t^{\top}h_t=1.
$$

定义一般化 Recall Rule：

$$
S_t^{\mathrm{rec}}
=
D_tS_{t-1}
+
\gamma_th_te_t^{\top}.
$$

代入 $h_t$ 与 $e_t$ 后，恰好得到

$$
S_t^{\mathrm{rec}}
=
\left(
D_t+
\gamma_t\frac{r_tr_t^{\top}}{\delta_t}
\right)S_{t-1}.
$$

#### 8.9.2 精确读出插值

由于

$$
q_t^{\top}r_t=\delta_t,
$$

所以

$$
q_t^{\top}D_t^{(q)}
=
q_t^{\top}D_t
+
\gamma_tr_t^{\top}.
$$

又因为

$$
r_t^{\top}=q_t^{\top}(I-D_t),
$$

故

$$
\boxed{
q_t^{\top}D_t^{(q)}S
=
(1-\gamma_t)q_t^{\top}D_tS
+
\gamma_tq_t^{\top}S
}.
$$

当 $\gamma_t=1$ 时，

$$
q_t^{\top}D_t^{(q)}S=q_t^{\top}S.
$$

由于 $D_t^{(q)}$ 对称，此时还可推出

$$
D_t^{(q)}q_t=q_t.
$$

#### 8.9.3 收缩性证明

令

$$
\xi_t=W_t^{1/2}q_t.
$$

因为

$$
\delta_t=\xi_t^{\top}\xi_t,
$$

所以

$$
\begin{aligned}
I-D_t^{(q)}
&=
W_t-
\gamma_t\frac{W_tq_tq_t^{\top}W_t}{q_t^{\top}W_tq_t}
\\
&=
W_t^{1/2}
\left(
I-
\gamma_t
\frac{\xi_t\xi_t^{\top}}{\xi_t^{\top}\xi_t}
\right)
W_t^{1/2}.
\end{aligned}
$$

当 $0\leq\gamma_t\leq1$ 时，中间矩阵半正定，因此

$$
I-D_t^{(q)}\succeq0.
$$

另一方面，校正项本身半正定，所以

$$
D_t^{(q)}\succeq D_t\succeq0.
$$

最终得到

$$
\boxed{
0\preceq D_t
\preceq D_t^{(q)}
\preceq I
}.
$$

因此

$$
\left\lVert D_t^{(q)}\right\rVert_2\leq1.
$$

这给出了 GDN-2 上应优先采用的稳定推广。

#### 8.9.4 标量情形是其特例

若

$$
D_t=\alpha_tI,
$$

则

$$
r_t=(1-\alpha_t)q_t,
$$

$$
\delta_t=1-\alpha_t.
$$

于是

$$
\frac{r_tr_t^{\top}}{\delta_t}
=
(1-\alpha_t)q_tq_t^{\top},
$$

从而恢复主公式

$$
D_t^{(q)}
=
\alpha_tI
+
\gamma_t(1-\alpha_t)q_tq_t^{\top}.
$$

### 8.10 GDN-2 的完整更新

若 GDN-2 的原始编辑写为

$$
S_t
=
\left(I-k_ta_t^{\top}\right)
D_tS_{t-1}
+
k_tz_t^{\top},
$$

则 QGDN 扩展为

$$
\boxed{
S_t
=
\left(I-k_ta_t^{\top}\right)
D_t^{(q)}S_{t-1}
+
k_tz_t^{\top}
}.
$$

这里 $a_t$ 是 GDN-2 原有编辑地址，与上一节的恢复地址 $h_t$ 不同。

稳定 Recall 算子只能保证

$$
\lVert D_t^{(q)}\rVert_2\leq1.
$$

若原始 GDN-2 编辑因子 $I-k_ta_t^{\top}$ 非正规，则完整传播矩阵仍只有

$$
\lVert A_t\rVert_2
\leq
\left\lVert I-k_ta_t^{\top}\right\rVert_2.
$$

所以 QGDN 不会修复 GDN-2 原有编辑算子的全部稳定性问题。

### 8.11 非对称简式只作为消融项

一个直观但不推荐作为主方法的公式是

$$
D_{t,\mathrm{asym}}^{(q)}
=
D_t+
\gamma_tP_t(I-D_t).
$$

它同样满足读出插值恒等式

$$
q_t^{\top}D_{t,\mathrm{asym}}^{(q)}S
=
(1-\gamma_t)q_t^{\top}D_tS
+
\gamma_tq_t^{\top}S.
$$

但当 $D_t$ 不是标量矩阵时，它通常不对称，也不保证谱范数不超过 $1$。

例如取

$$
D=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix},
\qquad
q=\frac{1}{\sqrt2}
\begin{bmatrix}
1\\
1
\end{bmatrix},
\qquad
\gamma=1,
$$

则

$$
D_{\mathrm{asym}}^{(q)}
=
\begin{bmatrix}
1&1/2\\
0&1/2
\end{bmatrix},
$$

其谱范数约为

$$
1.144>1.
$$

甚至两个单步看似温和的非正规算子交替时，也可能产生乘积增长。例如适当选择两个对角 $D_1,D_2$ 与两个旋转查询，可使

$$
\rho
\left(
D_{2,\mathrm{asym}}^{(q)}
D_{1,\mathrm{asym}}^{(q)}
\right)
\approx1.079>1.
$$

因此该简式适合作为以下消融：

- 是否只要读出插值就足够；
- 对称化与归一化是否确实改善训练稳定性；
- 非正规动力学是否在长序列上造成可观测退化。

### 8.12 低秩实现

对稳定的一般形式，令

$$
u_t=
\frac{r_t}{\sqrt{\delta_t}},
$$

则

$$
D_t^{(q)}=D_t+\gamma_tu_tu_t^{\top}.
$$

因此 Recall Rule 只增加一个秩一校正：

$$
D_t^{(q)}S
=
D_tS
+
\gamma_tu_t(u_t^{\top}S).
$$

对于 GDN-2，设

$$
c_t=a_t^{\top}u_t,
$$

则

$$
\begin{aligned}
A_t
&=
\left(I-k_ta_t^{\top}\right)
\left(D_t+\gamma_tu_tu_t^{\top}\right)
\\
&=
D_t-U_tV_t^{\top},
\end{aligned}
$$

其中

$$
U_t=
\begin{bmatrix}
u_t&k_t
\end{bmatrix},
$$

$$
V_t=
\begin{bmatrix}
-\gamma_tu_t
&
D_ta_t+\gamma_tc_tu_t
\end{bmatrix}.
$$

因此完整齐次更新仍是“对角加秩二”，保留了并行扫描或分块算法可利用的低秩结构。

### 8.13 $\gamma_t$ 的参数化

推荐将 $\gamma_t$ 设为每个 token、每个 head 的标量：

$$
\gamma_t
=
\sigma(w_\gamma^{\top}x_t+b_\gamma).
$$

可以比较三种设置：

1. 固定常数 $\gamma$；
2. 每个 head 一个可学习常数；
3. 每个 token、每个 head 的动态标量。

不建议在主版本中直接令 $\gamma_t$ 为逐维向量，因为这会重新引入坐标依赖，并破坏当前清楚的投影几何与稳定性证明。若要研究向量门，应明确将其定义为另一种模型变体。

---

## 9. Remarks and Interpretation

### 9.1 QGDN 保护的到底是什么

QGDN 保护的是当前查询对应的**可读子空间**，而不是状态矩阵中的某些固定行。对标量版本，该子空间就是

$$
\operatorname{span}(q_t).
$$

对一般对角衰减，实际校正方向变为

$$
(I-D_t)q_t,
$$

因为只有这一部分真正受到了基础衰减。

### 9.2 Recall Rule 与 Delta Rule 的关系

两者都具有“地址向量乘内容误差”的秩一结构：

$$
\text{state}
\leftarrow
\text{state}
+
\text{address}\times\text{error}^{\top}.
$$

区别在于：

- Delta Rule 面向新信息写入与旧关联修改；
- Recall Rule 面向基础遗忘造成的旧读出损失；
- 前者是外部目标驱动，后者是状态自监督的读出恢复。

因此论文中可以把 Recall Rule 描述为与 Delta Rule 互补的记忆操作，但不应把它重新命名为一种 Delta Rule。

### 9.3 为什么名称仍然保留 QGDN

“Recall Rule”描述局部更新机制；“QGDN”描述采用该机制的完整网络。标题

> **QGDN: A Query-Guided Recall Rule for Gated Delta Networks**

同时保留了模型家族、核心机制和应用对象，语义比强行让缩写对应 “Recall” 更自然。

### 9.4 推荐的研究叙事

论文应优先讲清以下因果链：

1. GDN 的基础衰减与当前访问需求无关；
2. 当前查询能够直接指出此刻需要读取的地址方向；
3. Recall Rule 恢复基础衰减刚刚损失的该方向读出；
4. 该恢复是秩一、可控并且在主版本中非扩张；
5. 随后仍由原始 Delta Rule 完成新信息编辑。

---

## 10. Boundaries and Non-Claims

本文不作以下过强声明：

1. **不声称保护全部旧记忆。** QGDN 只偏向当前查询访问的方向。
2. **不声称 $\gamma_t=1$ 时完整一步更新保持查询读出不变。** 保证只覆盖 Recall 步，随后 Delta 编辑仍可改变它。
3. **不声称非扩张等于长期状态有界。** 持续写入仍可能在单位模态上累积。
4. **不声称对角行衰减一定优于标量衰减。** 它会改变历史 key 的方向，需要实验论证收益。
5. **不声称稳定 Recall 算子能修复 GDN-2 原编辑因子的所有非正规性。**
6. **不声称 $\operatorname{Diag}(q_t^2)$ 是投影形式的廉价等价物。** 它对应不同的坐标门控归纳偏置。
7. **不声称状态梯度非爆炸就意味着所有参数梯度都不会累积。** 参数对多时刻门值的贡献仍可能相加。

---

## 11. Open Risks and Required Ablations

### 11.1 主要风险

- $\gamma_t$ 长期饱和到 $1$，导致有效遗忘不足；
- 连续查询高度相似，使相同记忆方向反复获得保护；
- 当前查询噪声较大时，模型可能保护错误方向；
- GDN-2 的原始非正规编辑因子可能掩盖 Recall 算子的稳定性收益；
- 当 $\delta_t$ 极小时，数值实现需要稳定分支；
- Recall 对旧读出的偏好可能妨碍必要的快速覆盖。

### 11.2 必需消融

至少比较：

1. 原始 GDN；
2. 标量 QGDN；
3. GDN-2；
4. 稳定对称 GDN-2 + Recall Rule；
5. 非对称简式；
6. $q_tq_t^{\top}$ 与 $\operatorname{Diag}(q_t^2)$；
7. 固定、逐头、逐 token 的 $\gamma_t$；
8. Recall 在 Delta 编辑之前与之后的顺序；
9. 对 $\gamma_t$ 加遗忘裕度正则与不加正则；
10. 长序列上的状态范数、梯度范数与有效记忆寿命。

### 11.3 数值实现建议

对一般稳定形式，若

$$
\delta_t=q_t^{\top}(I-D_t)q_t
$$

低于实现阈值，则直接令

$$
D_t^{(q)}=D_t.
$$

不建议无条件把分母替换为 $\delta_t+\varepsilon$ 后仍声称精确读出插值，因为这会改变恒等式。实现中应区分：

- 数学定义中的 $\delta_t=0$ 分支；
- 浮点计算中的近零稳定策略；
- 近零近似对读出恢复精度造成的误差。

---

## 12. Canonical Definition

论文主文中建议使用以下最简定义。

基础衰减：

$$
\widetilde S_t=\alpha_tS_{t-1}.
$$

Recall Rule：

$$
\boxed{
S_t^{\mathrm{rec}}
=
\widetilde S_t
+
\gamma_tq_t
\left(
S_{t-1}^{\top}q_t-
\widetilde S_t^{\top}q_t
\right)^{\top}
}.
$$

等价保持算子：

$$
\boxed{
D_t^{(q)}
=
\alpha_tI
+
\gamma_t(1-\alpha_t)q_tq_t^{\top}
}.
$$

完整 QGDN 更新：

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

一句话概括：

> QGDN lets the current query recall the portion of its old readout that would otherwise be erased by global decay, before the standard Delta update edits the memory.
