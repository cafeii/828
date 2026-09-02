# QR-GDN 双通道设计

令

$$
B_t^{KV}=\alpha_t^{KV}M_{t-1}^{KV},
\qquad
B_t^{QR}=\alpha_t^{QR}M_{t-1}^{QR},
$$

以及

$$
r_t=(M_{t-1}^{KV})^\top q_t.
$$

联合在线目标为

$$
\begin{aligned}
\min_{M^{KV},M^{QR}}\quad
&\frac12\|M^{KV}-B_t^{KV}\|_F^2
+\frac12\|M^{QR}-B_t^{QR}\|_F^2\\
&+\frac{\rho_t^{KV}}2\|(M^{KV})^\top k_t-v_t\|^2
+\frac{\rho_t^{QR}}2\|(M^{QR})^\top q_t-r_t\|^2,
\end{aligned}
$$

其中 $\rho_t^x=\beta_t^x/(1-\beta_t^x)$。归一化地址下的闭式解为

$$
M_t^{KV}
=B_t^{KV}
+\beta_t^{KV}k_t\left(v_t-(B_t^{KV})^\top k_t\right)^\top,
$$

$$
M_t^{QR}
=B_t^{QR}
+\beta_t^{QR}q_t\left(r_t-(B_t^{QR})^\top q_t\right)^\top.
$$

对应的耦合仿射递推是

$$
M_t^{KV}
=\alpha_t^{KV}(I-\beta_t^{KV}k_tk_t^\top)M_{t-1}^{KV}
+\beta_t^{KV}k_tv_t^\top,
$$

$$
M_t^{QR}
=\alpha_t^{QR}(I-\beta_t^{QR}q_tq_t^\top)M_{t-1}^{QR}
+\beta_t^{QR}q_tq_t^\top M_{t-1}^{KV}.
$$

因此联合状态具有块下三角转移：KV 通道接收外部写入，QR 通道接收 KV 通道中被实际查询访问的部分。

读出采用

$$
o_t=(M_t^{KV})^\top q_t
+\tanh(\ell_t)(M_{t-1}^{QR})^\top q_t,
$$

其中 $\ell_t$ 由当前输入生成且投影权重零初始化。这样保留现有 GDN 的 KV 读出语义，QR 写入只影响未来 token，并在初始化时严格退化为 GDN。
