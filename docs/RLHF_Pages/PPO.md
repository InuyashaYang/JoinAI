

# **RLHF (Reinforcement Learning from Human Feedback)**

在RLHF的训练过程中，模型通过人类反馈优化其行为。主要包括**奖励模型（Reward Model, RM）阶段**和**PPO阶段**。

## **1. 奖励模型（RM）阶段**

### **目标**
奖励模型根据人类反馈评估候选答案的质量，输出一个概率分布，反映不同答案的相对优劣。

### **概率分布构建**
给定输入 $x$ 和候选答案集合 $\{y_i\}_{i=1}^K$，奖励模型 $R(x, y)$ 生成评分。使用Softmax函数将评分转化为概率：
$$
P(y_i \mid x) = \frac{e^{R(x, y_i)}}{\sum_{j=1}^{K} e^{R(x, y_j)}}
$$

### **交叉熵损失函数**
训练奖励模型，使其预测的概率分布尽可能接近人类标注的分布。损失函数定义为：
$$
L = \mathbb{E}_{(x, \{y_i\}, b) \sim D}\left[-\log\left(\frac{e^{R(x, y_b)}}{\sum_{i=1}^{K} e^{R(x, y_i)}}\right)\right]
$$

### **经验风险近似**
在实际中使用有限训练数据集进行近似：
$$
L_{empirical} = -\frac{1}{N} \sum_{n=1}^{N} \log\left(\frac{e^{R(x_n, y_b^n)}}{\sum_{i=1}^{K_n} e^{R(x_n, y_i^n)}}\right)
$$

### **梯度计算**
损失函数对参数 $\theta$ 的梯度为：
$$
\nabla_{\theta} L_{empirical} = -\frac{1}{N} \sum_{n=1}^{N} \left[\nabla_{\theta} R(x_n, y_b^n; \theta) - \sum_{i=1}^{K_n} P(y_i^n \mid x_n) \nabla_{\theta} R(x_n, y_i^n; \theta)\right]
$$

## **2. PPO阶段**

在获得奖励模型后，使用PPO算法训练语言模型，以优化策略 $\pi_\theta$。

### **目标函数**
PPO的目标函数包含三部分：
1. **奖励相关项**：
   $$
   L_{RL} = \mathbb{E}_{\tau \sim \pi_\theta}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}R(s,a), \ \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon\right)R(s,a)\right)\right]
   $$

2. **KL散度惩罚项**：
   $$
   L_{KL} = \beta \cdot \text{KL}(\pi_{\theta_{old}} \parallel \pi_\theta)
   $$

3. **值函数损失**：
   $$
   L_V = \mathbb{E}_{\tau \sim \pi_\theta}\left[(V_\theta(s) - R_t)^2\right]
   $$

综合目标函数为：
$$
L = L_{RL} - L_{KL} - c_1 L_V
$$

### **训练步骤**
1. **数据收集**：使用当前策略 $\pi_\theta$ 生成样本轨迹 $\tau$。
2. **优势估计**：计算优势函数 $A(s,a)$。
3. **多次更新**：通过梯度上升/下降优化策略参数，更新 $\pi_\theta$。
4. **迭代**：重复上述步骤，直到策略收敛。

### **关键参数**
- $\epsilon$：裁剪参数，控制策略更新幅度。
- $\beta$：KL散度的权重，平衡新旧策略的差异。
- $c_1$：值函数损失的权重，平衡策略优化与值函数学习。

### **注意事项**
- **奖励函数设计**：确保奖励模型准确反映人类偏好。
- **稳定性控制**：通过KL散度惩罚和裁剪参数，避免策略更新过大导致不稳定。
- **采样效率**：优化数据采集和利用，提升训练效率。

---

## **3. 关键数学概念**

### **Softmax函数**
将评分转化为概率分布：
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

### **交叉熵损失**
衡量预测分布与真实分布的差异：
$$
H(P, Q) = -\sum_{i=1}^{K} P(i) \log Q(i)
$$
在RM中，真实分布为one-hot形式，损失简化为：
$$
H(P, Q) = -\log Q(b)
$$

### **最大化似然估计（MLE）**
优化目标为最大化最佳答案的似然：
$$
\mathcal{L}(\theta) = \prod_{n=1}^{N} P(y_b^n \mid x_n; \theta)
$$
等价于最小化负对数似然：
$$
-\log \mathcal{L}(\theta) = -\sum_{n=1}^{N} \log P(y_b^n \mid x_n; \theta)
$$

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