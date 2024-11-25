
[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

## GRPO的损失函数

### 损失函数的定义

GRPO 旨在优化策略在不同群组上的表现，使其在最不利的群组上也能保持良好的性能。为此，GRPO 将优化问题建模为一个 **最小-最大优化问题**，其损失函数定义为：

$
\min_{\theta \in \Theta} \max_{\alpha \in \Delta_K} \sum_{g=1}^{K} \alpha_g \cdot \mathcal{L}_{\text{DPO}}(\pi_{\theta}, \mathcal{T}_g)
$

其中：
- $\theta \in \Theta \subset \mathbb{R}^d$ 是策略 $\pi_{\theta}$ 的参数，$\Theta$ 是参数的可行空间。
- $\alpha = (\alpha_1, \alpha_2, \dots, \alpha_K) \in \Delta_K$ 是群组权重向量，$\Delta_K$ 为 $K$ 维单纯形：
  
  $
  \Delta_K = \left\{ \alpha \in \mathbb{R}^K \ \bigg| \ \alpha_g \geq 0 \ \forall g \in \{1, \dots, K\}, \ \sum_{g=1}^{K} \alpha_g = 1 \right\}
  $
  
- $\mathcal{L}_{\text{DPO}}(\pi_{\theta}, \mathcal{T}_g)$ 是策略 $\pi_{\theta}$ 在群组 $g$ 上的 **DPO 损失**，具体定义为：
  
  $
  \mathcal{L}_{\text{DPO}}(\pi_{\theta}; (x_g, y_w, y_l)) = \log\left( \sigma\left( \beta h_{\pi_{\theta}}(x_g, y_w, y_l) \right) \right)
  $
  
  其中：
  - $\sigma$ 是 Sigmoid 函数。
  - $\beta$ 是一个超参数，用于控制函数的陡峭程度。
  - $h_{\pi_{\theta}}(x_g, y_w, y_l)$ 是策略对样本 $(x_g, y_w, y_l)$ 的评分函数。

### 损失函数的推导过程

GRPO 的损失函数源于以下几点考虑：

1. **群组鲁棒性**：希望策略在所有群组上表现均衡，尤其是在表现较差的群组上提升其性能。
2. **最坏情况优化**：通过最小化在最不利群组上的损失，确保策略在任何群组上都不会有较大损失。
3. **加权优化**：引入群组权重 $\alpha$，动态调整各群组的权重，使得表现较差的群组获得更高的权重，从而在优化过程中得到更多关注。

具体推导步骤如下：

1. **定义优化目标**：

   我们希望找到一个策略参数 $\theta$，使得在所有群组上的最大损失最小。数学表达为：

   $
   \min_{\theta \in \Theta} \max_{g \in \{1, \dots, K\}} \mathcal{L}_{\text{DPO}}(\pi_{\theta}, \mathcal{T}_g)
   $
   
2. **引入群组权重 $\alpha$**：

   为了将多个群组的损失统一考虑，引入权重向量 $\alpha$，其中每个 $\alpha_g$ 表示群组 $g$ 的重要性。优化目标转化为加权损失的最大化：

   $
   \min_{\theta \in \Theta} \max_{\alpha \in \Delta_K} \sum_{g=1}^{K} \alpha_g \cdot \mathcal{L}_{\text{DPO}}(\pi_{\theta}, \mathcal{T}_g)
   $
   
3. **解释加权损失**：

   通过调整权重 $\alpha$，优化过程能够自适应地关注那些在当前策略下表现较差的群组。具体来说：
   - 若某群组 $g$ 的损失较高，优化过程中 $\alpha_g$ 会增加，促使策略参数 $\theta$ 更关注该群组的损失优化。
   - 由于 $\alpha$ 受到单纯形约束（权重和为1），因此总损失在各群组间平衡。
   
4. **DPO损失的设计**：

   DPO（Distributionally Robust Policy Optimization）损失函数设计为：

   $
   \mathcal{L}_{\text{DPO}}(\pi_{\theta}; (x_g, y_w, y_l)) = \log\left( \sigma\left( \beta \left[ r_{\theta}(x_g, y_w) - r_{\theta}(x_g, y_l) \right] \right) \right)
   $
   
   其中：
   - $r_{\theta}(x, y)$ 表示策略 $\pi_{\theta}$ 在输入 $x$ 下对响应 $y$ 的评分。
   - $\beta$ 控制函数的敏感度，使得评分差异更为明显时，损失变化更剧烈。
   
   该设计的目的是鼓励策略在正确响应 $y_w$ 上的评分高于错误响应 $y_l$，通过 Sigmoid 函数平滑地衡量这种差异。

### 损失函数的优化机制

在损失函数的优化过程中，GRPO 采用 **交替优化** 的方式，即在每一轮迭代中，交替更新策略参数 $\theta$ 和群组权重 $\alpha$：

1. **固定策略参数 $\theta$，优化群组权重 $\alpha$**：
   - 通过最大化加权损失，确定当前策略下最需要关注的群组。

2. **固定群组权重 $\alpha$，优化策略参数 $\theta$**：
   - 通过最小化加权损失，优化策略在重点关注的群组上的表现。

这种交替优化的机制确保了策略在每一轮迭代中既能关注当前表现较差的群组，又能整体上提升所有群组的性能。

## GRPO的流程

在理解了 GRPO 的损失函数及其优化机制后，接下来介绍 GRPO 的具体流程步骤。

### 1. 初始化

- **步长参数**：
  - 群组权重更新步长 $\eta_{\alpha}$。
  - 策略参数更新步长 $\eta_{\theta}$。
  
- **初始权重**：
  - 策略的初始参数 $\theta^{(0)}$。
  - 每个群组的初始权重 $\alpha^{(0)}$（通常初始化为均匀分布，即 $\alpha_g^{(0)} = \frac{1}{K}$）。
  
- **投影算子**：
  - $\mathrm{P}_{\Theta}$，用于确保更新后的 $\theta$ 仍在参数空间 $\Theta$ 内。

### 2. 输入参数

- **数据集**：
  - $\mathcal{T}$，总样本数量为 $N = |\mathcal{T}|$。
  - 数据集被划分为 $K$ 个群组，每个群组的大小为 $N_g$，其中 $g \in \{1, 2, \dots, K\}$。

- **损失函数**：
  - $l(\pi_{\theta}; \cdot)$，用于评估策略的表现，这里指的是 DPO 损失。

### 3. 迭代更新（重复 $T$ 次）

对每一轮迭代 $t = 1, \dots, T$，执行以下步骤：

#### a. 复制当前群组权重

$
\alpha' \leftarrow \alpha^{(t-1)}
$

- 创建一个临时的群组权重向量 $\alpha'$，用于在当前迭代中更新。

#### b. 采样群组和数据

- **采样群组**：
  
  从群组分布 $\mathrm{Categorical}\left(\frac{N_1}{N}, \frac{N_2}{N}, \dots, \frac{N_K}{N}\right)$ 中采样一个群组 $g$。这里，群组的采样概率与其在数据集中所占的比例成正比。

- **采样数据点**：
  
  从群组 $g$ 的数据集中采样一个数据点 $(x_g, y_w, y_l) \sim \mathcal{T}_g$，其中：
  - $x_g$ 是输入特征。
  - $y_w$ 是“赢”（正确）的响应标签。
  - $y_l$ 是“输”（错误）的响应标签。

#### c. 更新群组权重

$
\alpha'_g \leftarrow \alpha'_g \exp\left( \eta_{\alpha} \cdot \frac{N \cdot l(\pi_{\theta^{(t-1)}}; (x_g, y_w, y_l))}{N_g} \right)
$

- **指数加权**：
  
  根据当前策略 $\pi_{\theta^{(t-1)}}$ 在群组 $g$ 上的损失 $l(\pi_{\theta^{(t-1)}}; (x_g, y_w, y_l))$，对群组权重 $\alpha'_g$ 进行指数加权更新。
  
- **归一化因子**：
  
  因子 $\frac{N}{N_g}$ 用于归一化，确保群组大小对权重更新有适当影响，避免群组大小差异过大对优化过程产生不均衡影响。

#### d. 归一化群组权重

$
\alpha^{(t)} \leftarrow \frac{\alpha'}{\sum_{g'} \alpha'_{g'}}
$

- 将更新后的群组权重 $\alpha'$ 进行归一化，使得 $\alpha^{(t)}$ 成为一个位于单纯形 $\Delta_K$ 上的向量（即 $\sum_{g} \alpha_g^{(t)} = 1$）。
- 归一化确保群组权重在每一轮迭代后保持有效的概率分布。

#### e. 更新策略参数

$
\theta^{(t)} \leftarrow \mathrm{P}_{\Theta} \left( \theta^{(t-1)} - \eta_{\theta} \cdot \frac{N \alpha_g^{(t)} \nabla_{\theta} l(\pi_{\theta^{(t-1)}}; (x_g, y_w, y_l))}{N_g} \right)
$

- **加权梯度下降**：
  
  使用加权后的梯度 $\alpha_g^{(t)} \nabla_{\theta} l(\pi_{\theta^{(t-1)}}; (x_g, y_w, y_l))$ 对策略参数 $\theta$ 进行更新。
  
- **权重的作用**：
  
  权重 $\alpha_g^{(t)}$ 保证了在当前策略下表现较差的群组在策略更新时被给予更多关注。
  
- **归一化因子**：
  
  因子 $\frac{N}{N_g}$ 再次确保群组大小对更新的影响适度，避免小群组因样本量少而被忽视。
  
- **投影算子**：
  
  通过 $\mathrm{P}_{\Theta}$，确保更新后的策略参数 $\theta^{(t)}$ 仍然位于参数空间 $\Theta$ 内。

### 4. 返回最终策略

$
\textbf{Return:} \quad \pi(\theta^{(T)})
$

- 在完成 $T$ 轮迭代后，输出最终优化得到的策略 $\pi(\theta^{(T)})$。

## 损失函数的梯度推导

为了更清晰地理解策略参数更新的机制，以下对损失函数的梯度进行详细推导。

### 加权梯度

策略参数更新的关键在于对加权后的梯度进行下降优化：

$
\theta^{(t)} \leftarrow \mathrm{P}_{\Theta} \left( \theta^{(t-1)} - \eta_{\theta} \cdot \frac{N \alpha_g^{(t)} \nabla_{\theta} l(\pi_{\theta^{(t-1)}}; (x_g, y_w, y_l))}{N_g} \right)
$

其中，加权梯度为：

$
\alpha_g^{(t)} \nabla_{\theta} l(\pi_{\theta^{(t-1)}}; (x_g, y_w, y_l))
$

### 损失函数的具体梯度

根据 DPO 损失函数的定义：

$
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; (x_g, y_w, y_l)) = \log\left( \sigma\left( \beta \left[ r_{\theta}(x_g, y_w) - r_{\theta}(x_g, y_l) \right] \right) \right)
$

对 $\theta$ 求梯度：

$
\nabla_{\theta} \mathcal{L}_{\text{DPO}}(\pi_{\theta}; (x_g, y_w, y_l)) = \sigma\left( \beta \left[ r_{\theta}(x_g, y_w) - r_{\theta}(x_g, y_l) \right] \right) \cdot \beta \cdot \left[ \nabla_{\theta} r_{\theta}(x_g, y_w) - \nabla_{\theta} r_{\theta}(x_g, y_l) \right]
$

结合权重 $\alpha_g^{(t)}$，加权梯度为：

$
\alpha_g^{(t)} \nabla_{\theta} \mathcal{L}_{\text{DPO}}(\pi_{\theta}; (x_g, y_w, y_l)) = \alpha_g^{(t)} \cdot \sigma\left( \beta \left[ r_{\theta}(x_g, y_w) - r_{\theta}(x_g, y_l) \right] \right) \cdot \beta \cdot \left[ \nabla_{\theta} r_{\theta}(x_g, y_w) - \nabla_{\theta} r_{\theta}(x_g, y_l) \right]
$

### 梯度的直观解释

- **Sigmoid函数 $\sigma$ 的作用**：
  
  $\sigma\left( \beta \left[ r_{\theta}(x_g, y_w) - r_{\theta}(x_g, y_l) \right] \right)$ 衡量了当前策略在群组 $g$ 上对正确响应 $y_w$ 和错误响应 $y_l$ 的评分差异。越大的差异意味着策略在该样本上的表现越好，损失越低。

- **权重 $\alpha_g^{(t)}$ 的调节**：
  
  当某群组的损失较高时，$\alpha_g^{(t)}$ 增大，促使梯度更新时更多地关注该群组的样本，从而优化策略在该群组上的表现。

- **梯度方向**：
  
  梯度 $\left[ \nabla_{\theta} r_{\theta}(x_g, y_w) - \nabla_{\theta} r_{\theta}(x_g, y_l) \right]$ 表示增加对正确响应 $y_w$ 的评分，同时减少对错误响应 $y_l$ 的评分。这有助于策略更准确地做出正确决策。

## 总结

GRPO 通过构建一个最小-最大优化的损失函数，结合加权梯度下降和群组权重动态调整机制，实现了策略在不同群组上的鲁棒性优化。其核心优势体现在：

1. **群组鲁棒性**：确保策略在所有群组上都有良好的表现，尤其是在表现较差的群组上有针对性的提升。
2. **自适应加权**：通过动态调整群组权重，使得优化过程能自适应地聚焦于当前最需要改进的群组。
3. **梯度优化**：通过精心设计的损失函数和梯度更新机制，确保策略参数向着最优方向快速收敛。
4. **理论保障**：在一定的假设条件下（如损失函数的凸性和 Lipschitz 连续性），GRPO 的优化过程能够保证以 $\mathcal{O}(T^{-1/2})$ 的速率收敛到最优解。

<script src="https://giscus.app/client.js"
        data-repo="InuyashaYang/AIDIY"
        data-repo-id="R_kgDOM1VVTQ"
        data-category="Announcements"
        data-category-id="DIC_kwDOM1VVTc4Ckls_"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="zh-CN"
        crossorigin="anonymous"
        async>
</script>