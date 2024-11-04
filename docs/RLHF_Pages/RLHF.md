[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

# RLHF (Reinforcement Learning from Human Feedback) 阶段


## RM阶段

RM使用交叉熵损失是为了将奖励模型的输出转化为概率分布，从而能够比较不同答案的相对质量，并通过最大化最佳答案的概率来学习人类的偏好。

对于给定的数据分布 $D$，损失函数的期望为：

$
L = \mathbb{E}_{(x, \{y_i\}_{i=1}^{K}, b) \sim D}\left[-\log\left(\frac{e^{R(x, y_b)}}{\sum_{i=1}^{K} e^{R(x, y_i)}}\right)\right]
$

这里，$\mathbb{E}$ 表示期望，$K$ 是候选答案的数量，$b$ 是最佳答案的索引。

在实践中，我们通常使用经验风险（empirical risk）来近似这个期望，即在有限的训练数据集上计算平均损失：

$
L_{empirical} = -\frac{1}{N} \sum_{n=1}^{N} \log\left(\frac{e^{R(x_n, y_b^n)}}{\sum_{i=1}^{K_n} e^{R(x_n, y_i^n)}}\right)
$

其中：
- $N$ 是训练数据集的大小。
- $K_n$ 是第 $n$ 个样本的候选答案数量，这允许每个样本有不同数量的候选答案。
- $x_n$ 是第 $n$ 个样本的问题。
- $y_i^n$ 是第 $n$ 个样本的第 $i$ 个候选答案。
- $y_b^n$ 是第 $n$ 个样本的最佳答案。




[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)
