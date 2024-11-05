[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

# RLHF (Reinforcement Learning from Human Feedback) 
在RLHF的训练过程中,模型通过人类反馈来优化其行为。整个过程通常包括多个阶段,其中**奖励模型（Reward Model, RM）阶段**是关键的一步,用于学习人类偏好,从而为后续的强化学习阶段提供指导。
## RM阶段
### 奖励模型的目标
奖励模型的主要目标是根据人类反馈评估不同候选答案的质量。为了实现这一目标,我们需要将奖励模型的输出转化为一个概率分布,使其能够比较不同答案的相对优劣。这一过程通常通过 **交叉熵损失函数（Cross-Entropy Loss）** 来实现。
### 概率分布的构建
首先,假设我们有一个数据分布 $D$,其中每个样本由一个输入 $x$、一组候选答案 $\{y_i\}_{i=1}^K$ 以及一个标记最佳答案的索引 $b$ 组成。奖励模型 $R(x, y)$ 为每个候选答案 $y$ 赋予一个实数评分。
为了将这些评分转化为概率分布,我们使用**Softmax函数**,其定义为:
$$
P(y_i \mid x) = \frac{e^{R(x, y_i)}}{\sum_{j=1}^{K} e^{R(x, y_j)}}
$$
这里,$P(y_i \mid x)$ 表示在给定输入 $x$ 的情况下,答案 $y_i$ 被选择的概率。Softmax函数确保所有概率值非负且总和为1,从而形成一个有效的概率分布。
### 交叉熵损失函数
为了训练奖励模型,使其能够准确反映人类偏好,我们采用交叉熵损失函数。交叉熵损失衡量的是两个概率分布之间的差异,这里我们希望模型预测的概率分布尽可能接近人类标注的分布。
对于给定的数据分布 $D$,损失函数的期望定义为:
$$
L = \mathbb{E}_{(x, \{y_i\}_{i=1}^{K}, b) \sim D}\left[-\log\left(\frac{e^{R(x, y_b)}}{\sum_{i=1}^{K} e^{R(x, y_i)}}\right)\right]
$$
#### 各符号解释：

- $\mathbb{E}$ 表示对数据分布 $D$ 上的期望。
- $K$ 是每个样本中候选答案的数量。
- $b$ 是标记为最佳答案的索引。
- $R(x, y_b)$ 是输入 $x$ 和最佳答案 $y_b$ 的奖励评分。
- $\sum_{i=1}^{K} e^{R(x, y_i)}$ 是所有候选答案奖励评分的指数和,用于归一化。

### 损失函数的直观理解
损失函数 $L$ 的目标是最大化最佳答案 $y_b$ 的概率,即最小化 $-\log P(y_b \mid x)$。通过最小化这个损失,奖励模型会调整其参数,使得最佳答案在给定输入下的概率尽可能高,从而更好地反映人类的偏好。
### 经验风险近似
在实际应用中,我们无法直接计算期望值 $\mathbb{E}$,因为它涉及到整个数据分布 $D$。因此,我们通常使用**经验风险（Empirical Risk）**来近似这个期望,即在有限的训练数据集上计算平均损失。
假设我们有一个包含 $N$ 个样本的训练数据集,每个样本的候选答案数量可以不同（记为 $K_n$,其中 $n = 1, 2, \dots, N$）。经验风险定义为:
$$
L_{empirical} = -\frac{1}{N} \sum_{n=1}^{N} \log\left(\frac{e^{R(x_n, y_b^n)}}{\sum_{i=1}^{K_n} e^{R(x_n, y_i^n)}}\right)
$$
#### 各符号解释：

- $N$ 是训练数据集的大小。
- $K_n$ 是第 $n$ 个样本的候选答案数量。
- $x_n$ 是第 $n$ 个样本的输入。
- $y_i^n$ 是第 $n$ 个样本的第 $i$ 个候选答案。
- $y_b^n$ 是第 $n$ 个样本的最佳答案。

### 损失函数的优化
通过最小化经验风险 $L_{empirical}$,我们实质上在进行**最大似然估计（Maximum Likelihood Estimation, MLE）**,即选择一组模型参数,使得最佳答案在训练数据上的似然最大化。这一过程通常通过梯度下降等优化算法来实现。
### 梯度计算
为了优化损失函数,我们需要计算其关于奖励模型参数 $\theta$ 的梯度。假设奖励模型 $R(x, y; \theta)$ 依赖于参数 $\theta$,则损失函数的梯度为:
$$
\nabla_{\theta} L_{empirical} = -\frac{1}{N} \sum_{n=1}^{N} \nabla_{\theta} \log P(y_b^n \mid x_n)
$$
其中,
$$
\log P(y_b^n \mid x_n) = R(x_n, y_b^n; \theta) - \log\left(\sum_{i=1}^{K_n} e^{R(x_n, y_i^n; \theta)}\right)
$$
进一步展开梯度:
$$
\nabla_{\theta} \log P(y_b^n \mid x_n) = \nabla_{\theta} R(x_n, y_b^n; \theta) - \sum_{i=1}^{K_n} P(y_i^n \mid x_n) \nabla_{\theta} R(x_n, y_i^n; \theta)
$$
这表明,优化过程中不仅要增强最佳答案的评分,还需要抑制其他候选答案的评分,使得最佳答案在概率分布中占据主导地位。
 
# 进一步的数学铺垫
为了全面理解RM阶段的工作原理,以下是一些关键的数学概念和推导:
## Softmax 函数
Softmax 函数是一种常见的激活函数,用于将一组实数转化为正数且和为1的概率分布。对于给定的向量 $\mathbf{z} = (z_1, z_2, \dots, z_K)$,Softmax 函数定义为:
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$
在奖励模型中,$\mathbf{z}$ 通常由奖励模型输出的评分 $\{R(x, y_i)\}_{i=1}^K$ 组成。
## 交叉熵损失函数
交叉熵用于衡量两个概率分布之间的差异。给定真实分布 $P$ 和预测分布 $Q$,交叉熵定义为:
$$
H(P, Q) = -\sum_{i=1}^{K} P(i) \log Q(i)
$$
在RM阶段,我们将真实分布 $P$ 设定为在最佳答案位置上为1,其余位置为0（即**one-hot 分布**）,这样交叉熵损失简化为:
$$
H(P, Q) = -\log Q(b)
$$
其中,$b$ 是最佳答案的索引,$Q(b) = P(y_b \mid x)$。
## 最大化似然估计（Maximum Likelihood Estimation, MLE）
MLE 的目标是找到一组参数,使得在给定数据的情况下,观测到的最佳答案的概率最大。这等价于最大化似然函数:
$$
\mathcal{L}(\theta) = \prod_{n=1}^{N} P(y_b^n \mid x_n; \theta)
$$
取对数并取负后,转化为最小化交叉熵损失:
$$
-\log \mathcal{L}(\theta) = -\sum_{n=1}^{N} \log P(y_b^n \mid x_n; \theta)
$$
因此,最小化负对数似然（即经验风险）与最大化对数似然是等价的优化目标。

## PPO阶段
 在获得奖励模型(RM)后,我们使用PPO (Proximal Policy Optimization)算法来训练语言模型。PPO的目标函数通常包含三个主要部分：

1. 奖励相关项：
$L_{RL} = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}R(s,a), \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon\right)R(s,a)\right) \right]$

2. KL散度惩罚项：
$L_{KL} = \beta \cdot \text{KL}(\pi_{\theta_{old}} || \pi_\theta)$

3. 值函数损失：
$L_V = \mathbb{E}_{\tau \sim \pi_\theta}\left[(V_\theta(s) - R_t)^2\right]$

总的目标函数为：
$L = L_{RL} - L_{KL} - c_1L_V$

其中：

- $\pi_\theta$ 是当前策略
- $\pi_{\theta_{old}}$ 是旧策略
- $R(s,a)$ 是奖励函数
- $\epsilon$ 是裁剪参数
- $\beta$ 是KL散度的权重
- $c_1$ 是值函数损失的权重

主要训练步骤：

1. 使用当前策略收集数据
2. 计算优势估计
3. 多次更新策略参数
4. 重复以上步骤直到收敛

关键考虑点：
- 合理设计奖励函数
- 控制KL散度以保持稳定性
- 平衡探索与利用
- 注意采样效率


[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)
