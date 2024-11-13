# DPO Exercice


### 练习题 1: 偏好分数计算

**问题：**

给定以下概率值：

- $\log \pi_\theta(y_w|x) = -2.0$
- $\log \pi_\text{ref}(y_w|x) = -2.5$
- $\log \pi_\theta(y_l|x) = -3.0$
- $\log \pi_\text{ref}(y_l|x) = -3.5$

计算偏好分数 $r(x,y_w,y_l)$。

<details>
<summary>提示</summary>

使用定义：

$$
r(x,y_w,y_l) = \left[\log \pi_\theta(y_w|x) - \log \pi_\text{ref}(y_w|x)\right] - \left[\log \pi_\theta(y_l|x) - \log \pi_\text{ref}(y_l|x)\right]
$$

代入给定的值：

$$
r = (-2.0 - (-2.5)) - (-3.0 - (-3.5)) = (0.5) - (0.5) = 0
$$

</details>

---

### 练习题 2: 损失函数理解

**问题：**

假设计算得到的偏好分数为 $r = 1.2$。计算损失函数 $\mathcal{L}_\text{DPO}$ 的值。

<details>
<summary>提示</summary>

损失函数定义为：

$$
\mathcal{L}_\text{DPO} = -\log(\sigma(r))
$$

其中 $\sigma(r) = \frac{1}{1 + e^{-r}}$ 是 sigmoid 函数。

首先计算 $\sigma(1.2)$：

$$
\sigma(1.2) = \frac{1}{1 + e^{-1.2}} \approx 0.7685
$$

然后：

$$
\mathcal{L}_\text{DPO} = -\log(0.7685) \approx 0.263
$$

</details>

---

### 练习题 3: 梯度计算

**问题：**

假设偏好分数 $r = 0.8$，计算 $\mathcal{L}_\text{DPO}$ 关于 $r$ 的导数 $\frac{d\mathcal{L}_\text{DPO}}{dr}$。

<details>
<summary>提示</summary>

损失函数：

$$
\mathcal{L}_\text{DPO} = -\log(\sigma(r))
$$

首先，计算 $\sigma(r)$ 的导数：

$$
\frac{d\sigma(r)}{dr} = \sigma(r)(1 - \sigma(r))
$$

利用链式法则：

$$
\frac{d\mathcal{L}_\text{DPO}}{dr} = -\frac{1}{\sigma(r)} \cdot \sigma(r)(1 - \sigma(r)) = -(1 - \sigma(r))
$$

所以：

$$
\frac{d\mathcal{L}_\text{DPO}}{dr} = \sigma(r) - 1
$$

具体计算：

$$
\sigma(0.8) = \frac{1}{1 + e^{-0.8}} \approx 0.689974
$$

因此：

$$
\frac{d\mathcal{L}_\text{DPO}}{dr} \approx 0.689974 - 1 = -0.310026
$$

</details>

---

### 练习题 4: 参考模型的作用

**问题：**

解释在 DPO 训练过程中，参考模型 $\pi_\text{ref}$ 的作用是什么？为什么在训练时保持其参数固定？

<details>
<summary>提示</summary>

参考模型 $\pi_\text{ref}$ 作为一个固定的基准，提供了一个稳定的标准来比较政策模型 $\pi_\theta$ 的输出。这种设计有助于：

1. **正则化效果**：防止政策模型偏离参考模型过远，保持训练的稳定性。
2. **基线对比**：通过比较 $\pi_\theta$ 和 $\pi_\text{ref}$ 的概率，衡量政策模型在偏好优化中的改进。
3. **减少训练不稳定性**：固定参考模型的参数避免在训练过程中引入额外的变化，从而简化优化过程。

保持参考模型参数固定，确保其作为一个稳定的标准，不受政策模型更新的影响，从而为政策模型的优化提供一致的对比基础。

</details>

---

### 练习题 5: 自回归生成

**问题：**

什么是**自回归生成**？在 DPO 的前向传播过程中，自回归生成如何用于计算序列的条件概率？

<details>
<summary>提示</summary>

**自回归生成**是一种生成模型的方法，其中每一步的输出都依赖于之前生成的输出。具体来说，模型在生成序列时，每生成一个新的令牌，都会将之前生成的令牌作为输入的一部分，以此来预测下一个令牌的概率分布。

在 DPO 的前向传播过程中，自回归生成用于计算序列 $y$ 的条件概率：

$$
\pi_\theta(y|x) = \prod_{t=1}^T \pi_\theta(y^t|x, y^{<t})
$$

这种方式确保了序列中每个令牌的生成都是基于之前生成的内容，使得整体序列具有连贯性和一致性。

</details>

---

### 练习题 6: 实际概率计算

**问题：**

假设问题序列 $x$ 包含两个时间步，回答序列 $y_w$ 包含两个令牌 $y_w^1$ 和 $y_w^2$。给定以下条件概率：

- 政策模型：
  - $\pi_\theta(y_w^1|x) = 0.4$
  - $\pi_\theta(y_w^2|x, y_w^1) = 0.5$

- 参考模型：
  - $\pi_\text{ref}(y_w^1|x) = 0.3$
  - $\pi_\text{ref}(y_w^2|x, y_w^1) = 0.6$

计算：
1. $\log \pi_\theta(y_w|x)$
2. $\log \pi_\text{ref}(y_w|x)$
3. 偏好分数 $r(x,y_w,y_l)$，假设 $\log \pi_\theta(y_l|x) = -2.5$ 和 $\log \pi_\text{ref}(y_l|x) = -3.0$

<details>
<summary>提示</summary>

**1. 计算 $\log \pi_\theta(y_w|x)$：**

$$
\pi_\theta(y_w|x) = \pi_\theta(y_w^1|x) \cdot \pi_\theta(y_w^2|x, y_w^1) = 0.4 \times 0.5 = 0.2
$$

$$
\log \pi_\theta(y_w|x) = \log 0.2 \approx -1.6094
$$

**2. 计算 $\log \pi_\text{ref}(y_w|x)$：**

$$
\pi_\text{ref}(y_w|x) = \pi_\text{ref}(y_w^1|x) \cdot \pi_\text{ref}(y_w^2|x, y_w^1) = 0.3 \times 0.6 = 0.18
$$

$$
\log \pi_\text{ref}(y_w|x) = \log 0.18 \approx -1.7148
$$

**3. 计算偏好分数 $r(x,y_w,y_l)$：**

使用偏好分数的定义：

$$
r = \left[\log \pi_\theta(y_w|x) - \log \pi_\text{ref}(y_w|x)\right] - \left[\log \pi_\theta(y_l|x) - \log \pi_\text{ref}(y_l|x)\right]
$$

代入数值：

$$
r = (-1.6094 - (-1.7148)) - (-2.5 - (-3.0)) = (0.1054) - (0.5) = -0.3946
$$

</details>

---

### 练习题 7: 理解损失函数优化目标

**问题：**

在 DPO 中，损失函数 $\mathcal{L}_\text{DPO} = -\log(\sigma(r(x,y_w,y_l)))$ 被最小化。解释最小化这个损失函数对于偏好分数 $r(x,y_w,y_l)$ 有什么要求？即，$r$ 应如何变化以使损失最小化？

<details>
<summary>提示</summary>

损失函数：

$$
\mathcal{L}_\text{DPO} = -\log(\sigma(r))
$$

其中 $\sigma(r)$ 是 sigmoid 函数，范围在 $(0, 1)$。

要最小化 $\mathcal{L}_\text{DPO}$，需要最大化 $\log(\sigma(r))$，这相当于最大化 $\sigma(r)$。

因为 $\sigma(r)$ 随着 $r$ 的增加而增加（$\sigma(r) \to 1$ 当 $r \to \infty$），最小化损失函数要求偏好分数 $r$ 趋向于更大的正值。

具体来说：

- 当 $r$ 增加时，$\sigma(r)$ 增加，$\log(\sigma(r))$ 增加，因此 $-\log(\sigma(r))$ 减少。
- 反之，当 $r$ 减少时，$\sigma(r)$ 减少，$\log(\sigma(r))$ 减少，导致损失增加。

因此，优化目标要求 $r(x,y_w,y_l)$ 趋向于尽可能大的正值，从而使政策模型更倾向于生成较好的回答 $y_w$ 而非较差的回答 $y_l$。

</details>

---

### 练习题 8: 参数更新机制

**问题：**

在 DPO 训练过程中，只有政策模型的参数 $\theta$ 被更新，而参考模型的参数保持不变。讨论这种设计的优点和可能的局限性。

<details>
<summary>提示</summary>

**优点：**

1. **稳定性**：固定参考模型的参数提供了一个稳定的基准，避免在训练过程中参考模型的变化引入不稳定性。
2. **简化优化**：只需优化政策模型，减少了训练的复杂性和计算资源的需求。
3. **防止模式崩溃**：固定参考模型可以防止政策模型过度拟合或偏离合理的生成分布。

**可能的局限性：**

1. **参考模型过时**：如果参考模型的能力有限或与当前任务需求不匹配，可能限制政策模型的优化效果。
2. **缺乏适应性**：在动态环境中，固定参考模型可能无法适应新的数据模式或偏好变化。
3. **依赖参考模型质量**：政策模型的优化效果高度依赖于参考模型的质量和覆盖范围，如果参考模型表现不佳，可能影响最终结果。

</details>

---

### 练习题 9: 扩展思考

**问题：**

如果在训练数据中没有明确的较差回答 $y_l$，你如何修改 DPO 的训练流程来处理这种情况？

<details>
<summary>提示</summary>

缺少明确的较差回答时，可以采取以下几种方法：

1. **负采样（Negative Sampling）**：
   - 随机选择政策模型生成的其他回答作为较差回答 $y_l$。
   - 确保这些回答与较好回答 $y_w$ 在质量上有一定差异。

2. **对比学习（Contrastive Learning）**：
   - 创建对比对，其中较好回答与较差回答形成正负样本对。
   - 利用对比损失函数增强模型对好回答的偏好。

3. **生成较差回答**：
   - 使用参考模型或其他生成模型专门生成质量较低的回答作为 $y_l$。
   - 设计方法确保生成的 $y_l$ 符合“较差”的定义。

4. **使用启发式规则**：
   - 根据一定规则（如回答长度、复杂度、语法错误等）筛选或修改回答以生成较差回答。

5. **利用未标注数据中的负面示例**：
   - 从现有数据中提取自然存在的负面回答，作为 $y_l$。

6. **半监督学习**：
   - 结合少量标注的较差回答与大量未标注数据，通过半监督学习方法扩展较差回答的集合。

这些方法可以帮助在缺乏明确标注的较差回答时，依然有效地训练政策模型，提高其生成高质量回答的能力。

</details>
