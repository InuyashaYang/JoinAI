
## Negative Preference Optimization

负偏好优化（Negative Preference Optimization, $NPO$），是一种简单的GA损失的替代方案。$NPO$损失在高温度极限下退化为GA损失，但在任意有限温度下仍然下界稳定，而GA损失在该情况下则不具备这一性质。

$NPO$的灵感来源于偏好优化，并将其发展为仅使用{负样本}进行偏好优化的方法。

### 偏好优化

在偏好优化中，我们拥有带有偏好反馈的数据集 $\mathcal{D}_{\text{paired}} = \{ (\mathbf{x}_i, \mathbf{y}_{i, \text{good}}, \mathbf{y}_{i, \text{bad}})\}_{i \in [n]}$，其中 $(\mathbf{y}_{i, \text{good}}, \mathbf{y}_{i, \text{bad}})$ 是由预训练模型 $\pi_\theta$ 生成的两种响应，且人类比较后确定偏好关系 $\mathbf{y}_{i, \text{good}} \succ \mathbf{y}_{i, \text{bad}}$（此处“$\text{good}$”表示“获胜”，“$\text{bad}$”表示“失败”）。目标是使用 $\mathcal{D}_{\text{paired}}$ 对 $\pi_\theta$ 进行微调，使其更好地符合人类偏好。偏好优化的一种流行方法是直接偏好优化（Direct Preference Optimization, DPO），其目标函数为

$$
\mathcal{L}_{\text{DPO}, \beta}(\theta) = - \frac{1}{\beta} \mathbb{E}_{\mathcal{D}_{\text{paired}}}\left[ \log \sigma \left( \beta \log \frac{\pi_\theta(\mathbf{y}_{\text{good}} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_{\text{good}} \mid \mathbf{x})} - \beta \log \frac{\pi_\theta(\mathbf{y}_{\text{bad}} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_{\text{bad}} \mid \mathbf{x})} \right) \right].
$$

其中，$\sigma(t) = \frac{1}{1 + e^{-t}}$ 是sigmoid函数，$\beta > 0$ 是逆温度参数，$\pi_{\text{ref}}$ 是参考模型。

### 负偏好优化作为偏好优化

我们观察到，未学习（Unlearning）问题可以被归类为偏好优化框架的一种特殊情况，即每个 $(\mathbf{x}_i, \mathbf{y}_i) \in \mathcal{D}_{\text{FG}}$ 仅提供一个负响应 $\mathbf{y}_{i, \text{bad}} = \mathbf{y}_i$，而不提供任何正响应 $\mathbf{y}_{i, \text{good}}$。因此，在DPO的损失函数 (\ref{eqn.dpo}) 中，我们忽略 $\mathbf{y}_{\text{good}}$ 项，得到负偏好优化（Negative Preference Optimization, NPO）损失：

$$
\mathcal{L}_{\text{NPO}, \beta}(\theta) = - \frac{2}{\beta} \mathbb{E}_{\mathcal{D}_{\text{FG}}}\left[ \log \sigma\left( - \beta \log \frac{\pi_\theta(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} \right) \right] = \frac{2}{\beta} \mathbb{E}_{\mathcal{D}_{\text{FG}}}\left[ \log \left( 1 + \left( \frac{\pi_\theta(\mathbf{y} \mid \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})} \right)^\beta \right) \right].
$$

最小化 $\mathcal{L}_{\text{NPO}, \beta}$ 确保在遗忘集上的预测概率 $\pi_\theta(\mathbf{y}_i \mid \mathbf{x}_i)$ 尽可能小，从而实现未学习遗忘集的目标。

### 与梯度上升的关联

通过消除NPO损失中对数函数内的额外1（即将 $\log\left(1 + \left(\frac{\pi_\theta}{\pi_{\text{ref}}}\right)^\beta\right)$ 替换为 $\log\left(\left(\frac{\pi_\theta}{\pi_{\text{ref}}}\right)^\beta\right)$），我们可以从NPO损失恢复出GA损失。此外，我们证明了当 $\beta \to 0$ 时，NPO损失也会退化为GA损失，这表明NPO是GA的严格推广。

#### 命题 1（${NPO}$ 在 $\beta \to 0$ 时退化为 ${GA}$）

对于任意参数 $\theta$，有

$$
\lim_{\beta \to 0} \left( \mathcal{L}_{\text{NPO}, \beta}(\theta) - \frac{2}{\beta} \log 2 \right) = \mathcal{L}_{\text{GA}}(\theta) - \underbrace{\mathbb{E}_{\mathcal{D}_{\text{FG}}}\left[ \log \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x}) \right]}_{\text{与}~\theta~\text{无关}}.
$$

此外，假设 $\pi_\theta(\mathbf{y} \mid \mathbf{x})$ 对参数 $\theta$ 可微分，则

$$
\lim_{\beta \to 0} \nabla_\theta \mathcal{L}_{\text{NPO}, \beta}(\theta) = \nabla_\theta \mathcal{L}_{\text{GA}}(\theta).
$$


### $NPO$ 损失的稳定性

我们可以直观理解，$NPO$ 解决了GA损失在下界不稳定的问题。GA损失由于是交叉熵预测损失的负数，上界是无界的，可能导致训练过程中出现不稳定。而$NPO$损失在任意有限的 $\beta > 0$ 下都有下界，保证了训练的稳定性。

进一步地，$NPO$ 和 $GA$ 的梯度表现如下：

$$
\nabla_\theta \mathcal{L}_{\text{GA}} = \mathbb{E}_{\mathcal{D}_{\text{FG}}}\left[ \nabla_\theta \log \pi_\theta(\mathbf{y} \mid \mathbf{x}) \right],
$$

$$
\nabla_\theta \mathcal{L}_{\text{NPO}, \beta} = \mathbb{E}_{\mathcal{D}_{\text{FG}}}\left[ w_{\text{NPO}, \theta}(\mathbf{x}, \mathbf{y}) \nabla_\theta \log \pi_\theta(\mathbf{y} \mid \mathbf{x}) \right],
$$

其中，

$$
w_{\text{NPO}, \theta}(\mathbf{x}, \mathbf{y}) = \frac{2\pi_\theta^\beta(\mathbf{y} \mid \mathbf{x})}{\pi_\theta^\beta(\mathbf{y} \mid \mathbf{x}) + \pi_{\text{ref}}^\beta(\mathbf{y} \mid \mathbf{x})}
$$

可以被解释为一种自适应平滑权重。当样本 $(\mathbf{x}, \mathbf{y}) \in \mathcal{D}_{\text{FG}}$ 已经被有效遗忘，即 $\pi_\theta(\mathbf{y} \mid \mathbf{x}) \ll \pi_{\text{ref}}(\mathbf{y} \mid \mathbf{x})$，则 $w_{\text{NPO}, \theta}(\mathbf{x}, \mathbf{y}) \ll 1$，因此 $\|\nabla_\theta \mathcal{L}_{\text{NPO}, \beta}\|_2 \ll \|\nabla_\theta \mathcal{L}_{\text{GA}}\|_2$，从而 $NPO$ 的更新速度远低于 $GA$，避免了 $GA$ 可能出现的灾难性崩溃。

## 理论分析：发散速度

我们通过理论分析NPO和GA在标准逻辑回归设置下的发散速度，形式化上述直觉。在二分类问题中（$\mathbf{y} \in \{0,1\}$），考虑逻辑模型 $\pi_\theta(\mathbf{y}=1 \mid \mathbf{x}) = \sigma(\langle \mathbf{x}, \theta \rangle)$。初始模型记为 $\pi_{\theta_{\text{init}}}$，参数为 $\theta_{\text{init}} \in \mathbb{R}^d$。我们的目标是通过梯度下降，以步长 $\eta$ 对遗忘集 $\mathcal{D}_{\text{FG}} = \{ (\mathbf{x}_{\text{forget},i}, \mathbf{y}_{\text{forget},i}) \}_{i=1}^{n_{\text{forget}}}$ 最小化GA或NPO损失，进行 $T$ 次迭代。

### 定理 1（$GA$ 和 $NPO$ 的发散速度）

设 $\mathbf{X}_{\text{forget}} := (\mathbf{x}_{\text{forget},1}, \ldots, \mathbf{x}_{\text{forget},n_{\text{forget}}})^\top \in \mathbb{R}^{n_{\text{forget}} \times d}$。

考虑高维情形，其中 $n_{\text{forget}} \leq d$ 且假设 $\mathbf{X}_{\text{forget}} \mathbf{X}_{\text{forget}}^\top$ 可逆。

假设 $\|\theta_{\text{init}}\|_2 \leq R_0$，且对所有 $i \in [n_{\text{forget}}]$ 有 $\|\mathbf{x}_i\|_2 \in [r, R]$，其中 $R_0, r, R > 0$。

令 $\theta^{(t)}_{\text{GA}}$ 和 $\theta^{(t)}_{\text{NPO}}$ 分别表示对经验损失 $\mathcal{L}_{\text{GA}}$ 和 $\mathcal{L}_{\text{NPO}, \beta}$ 使用步长 $\eta$ 进行梯度下降的第 $t$ 次迭代。

- **$GA$ 线性发散**

  存在一些与 $(R_0, r, R)$ 相关的常数 $C_0, C_1, C_2 > 0$，当

  $$
  \max_{i \neq j} |\langle \mathbf{x}_i, \mathbf{x}_j \rangle| \leq \frac{C_0}{n_{\text{forget}}},
  $$

  时，对于任意 $t \geq 1$，有

  $$
  \|\theta^{(t)}_{\text{GA}} - \theta_{\text{init}}\|_{\mathbf{X}_{\text{forget}}^\top \mathbf{X}_{\text{forget}}} \in \left[ C_1 \cdot n_{\text{forget}}^{-1/2} \eta \cdot t, \ C_2 \cdot n_{\text{forget}}^{-1/2} \eta \cdot t \right].
  $$

- **$NPO$ 对数发散**

  假设 $\eta \leq 1$。存在一些与 $(R_0, r, R, \beta)$ 相关的常数 $C_0, C_1, C_2, C_3 > 0$，当

  $$
  \max_{i \neq j} | \langle \mathbf{x}_i, \mathbf{x}_j \rangle | \leq \frac{C_0}{n_{\text{forget}}},
  $$

  时，对于任意 $t \geq 1$，有

  $$
  \|\theta^{(t)}_{\text{NPO}} - \theta_{\text{init}}\|_{\mathbf{X}_{\text{forget}}^\top \mathbf{X}_{\text{forget}}} \in \left[ C_1 \sqrt{n_{\text{forget}}} \log\left(C_2 \cdot \eta \cdot n_{\text{forget}}^{-1} \cdot t + 1\right), \ C_3 \sqrt{n_{\text{forget}}} \log\left(C_3 \cdot \eta \cdot n_{\text{forget}}^{-1} \cdot t + 1\right) \right],
  $$
  
  其中 $C_3$ 出现在对数函数内部也是一个常数。
