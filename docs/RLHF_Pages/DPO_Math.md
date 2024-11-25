# 前置知识


## Bradley-Terry模型基本概念
Bradley-Terry模型是一个概率模型，用于描述配对比较(paired comparisons)中的偏好关系。其基本形式如下：
对于任意两个选项$i$和$j$，选择$i$优于$j$的概率为：
$P(i \succ j) = \frac{\pi_i}{\pi_i + \pi_j}$
其中：
- $\pi_i > 0$ 表示选项$i$的潜在"能力值"或"强度"
- $\succ$ 表示"优于"关系
## 数学性质
1) **对称性**：
```math
P(i \succ j) + P(j \succ i) = 1
```
2) **传递性**：
如果$P(i \succ j) > 0.5$且$P(j \succ k) > 0.5$，则$P(i \succ k) > 0.5$

## KL散度基本概念

KL散度（Kullback-Leibler散度）是一种用于衡量两个概率分布差异的度量。对于离散概率分布P和Q，KL散度定义如下：

```math
D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
```

其中：

- P(x)和Q(x)分别表示两个概率分布在事件x上的概率

- 对数通常采用自然对数（以e为底）

## 数学性质

1) **非负性**：
```math
D_{KL}(P||Q) \geq 0
```
当且仅当P和Q完全相同时，KL散度为0。

2) **不对称性**：
```math
D_{KL}(P||Q) \neq D_{KL}(Q||P)
```
KL散度不是一个对称的度量。

3) **凸性**：
KL散度关于其参数是凸函数。

4) **加性**：
对于独立随机变量，联合分布的KL散度等于边缘分布KL散度之和。

5) **连续性**：
对于连续概率分布，KL散度可以表示为积分形式：
```math
D_{KL}(P||Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx
```
# DPO符号表

| 符号 | 含义 |
|------|------|
| $\pi_\theta(y\|x)$ | 策略模型(policy model)的条件概率分布 |
| $\pi_\text{ref}(y\|x)$ | 参考模型(reference model)的条件概率分布 |
| $(x, y_w, y_l)$ | 偏好数据三元组，包含上下文x，获胜回复$y_w$和失败回复$y_l$ |
| $r_\theta(x,y)$ | 奖励模型(reward model)，定义为$\log(\pi_\theta(y\|x)) - \log(\pi_\text{ref}(y\|x))$ |
| $\beta$ | 温度参数，控制策略的保守程度 |

# DPO Loss推导

1) 首先，DPO的核心思想是将偏好学习转化为二分类问题。给定偏好对$(y_w, y_l)$，目标是最大化$y_w$被选中的概率。

2) 根据Bradley-Terry模型，我们可以写出获胜概率：

$P(y_w \succ y_l|x) = \frac{\exp(r_\theta(x,y_w))}{\exp(r_\theta(x,y_w)) + \exp(r_\theta(x,y_l))}$

3) 将奖励模型展开：

$r_\theta(x,y) = \log(\pi_\theta(y|x)) - \log(\pi_\text{ref}(y|x))$

4) DPO的损失函数可以写为：

$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\left(\frac{\exp(r_\theta(x,y_w)/\beta)}{\exp(r_\theta(x,y_w)/\beta) + \exp(r_\theta(x,y_l)/\beta)}\right)\right]$

# DPO Loss的数学性质

1) **单调性**：
- 当$\beta \to 0$时，loss趋向于硬分类损失
- 当$\beta \to \infty$时，模型行为趋近于参考模型

2) **梯度性质**：

首先回顾DPO的loss函数：

$\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\left(\frac{\exp(r_\theta(x,y_w)/\beta)}{\exp(r_\theta(x,y_w)/\beta) + \exp(r_\theta(x,y_l)/\beta)}\right)\right]$

让我们逐步计算梯度：

1) 为简化表示，令：

   - $r_w = r_\theta(x,y_w)/\beta$

   - $r_l = r_\theta(x,y_l)/\beta$

2) Loss可以重写为：
   $\mathcal{L} = -\log\left(\frac{\exp(r_w)}{\exp(r_w) + \exp(r_l)}\right)$
   $= -r_w + \log(\exp(r_w) + \exp(r_l))$

3) 计算梯度：
   $\nabla_\theta \mathcal{L} = -\nabla_\theta r_w + \frac{\exp(r_w)\nabla_\theta r_w + \exp(r_l)\nabla_\theta r_l}{\exp(r_w) + \exp(r_l)}$

4) 注意到：
   $p = \frac{\exp(r_l)}{\exp(r_w) + \exp(r_l)}$
   则$1-p = \frac{\exp(r_w)}{\exp(r_w) + \exp(r_l)}$

5) 代入整理得：
   $\nabla_\theta \mathcal{L} = -(1-p)\nabla_\theta r_w + p\nabla_\theta r_l$

6) 考虑$\beta$系数，最终得到：
   $\nabla_\theta \mathcal{L}_\text{DPO} = -\frac{1}{\beta}\mathbb{E}_{(x,y_w,y_l)}[(1-p)\nabla_\theta r_\theta(x,y_w) - p\nabla_\theta r_\theta(x,y_l)]$
## 2. DPO与KL散度的关系

DPO仍然与KL散度有密切关系，这体现在：

1) 回顾奖励定义：
   $r_\theta(x,y) = \log(\pi_\theta(y|x)) - \log(\pi_\text{ref}(y|x))$

2) 这实际上可以重写为：
   $r_\theta(x,y) = \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$

3) DPO的优化过程隐式地控制了策略$\pi_\theta$与参考策略$\pi_\text{ref}$之间的KL散度：
   $D_{KL}(\pi_\theta || \pi_\text{ref}) = \mathbb{E}_{y\sim\pi_\theta}[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}]$

### 优化过程分析

1) DPO的目标函数为：
   $L(\theta) = \mathbb{E}_{x,y_w,y_l}[\log\sigma(\frac{1}{\beta}(r_\theta(y_w|x) - r_\theta(y_l|x)))]$

2) 将$r_\theta$展开：
   $L(\theta) = \mathbb{E}_{x,y_w,y_l}[\log\sigma(\frac{1}{\beta}(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}))]$

3) 优化这个目标函数时，需要隐式地控制KL散度：


    KL散度的一般定义形式为：
   $D_{KL}(P||Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}$
    
    在DPO的上下文中：

   - $P$ 对应于策略 $\pi_\theta$

   - $Q$ 对应于参考策略 $\pi_\text{ref}$

   - 我们考虑的是条件概率分布（给定输入x时的输出y的分布）
    
    因此，将条件概率代入KL散度公式：
   $D_{KL}(\pi_\theta || \pi_\text{ref}) = \sum_y \pi_\theta(y|x)\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$

    当y是连续变量或者词表很大时，求和可以写作期望形式：
   $D_{KL}(\pi_\theta || \pi_\text{ref}) = \mathbb{E}_{y\sim\pi_\theta}[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}]$

    从而
    $D_{KL}(\pi_\theta || \pi_\text{ref}) = \mathbb{E}_{y\sim\pi_\theta}[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}]$
    这个公式实际上就是KL散度的标准定义在条件概率分布上的应用。

4) 温度参数$\beta$实际上在调节这个KL散度的约束强度：

   - 较小的$\beta$允许更大的KL散度

   - 较大的$\beta$限制策略偏离参考模型过远


    首先回顾DPO的目标函数：
   $L(\theta) = \mathbb{E}_{x,y_w,y_l}[\log\sigma(\beta(r_\theta(y_w|x) - r_\theta(y_l|x)))]$

    β的作用机制：

   - β乘以奖励差值：$\beta(r_\theta(y_w|x) - r_\theta(y_l|x))$

   - 当β较大时：

     - 即使奖励差值很小，乘以大的β后也会产生很大的梯度

     - 这使得优化过程更"谨慎"，因为小的策略偏差就会带来大的惩罚

     - 所以模型倾向于保持接近参考模型

   - 当β较小时：

     - 需要较大的奖励差值才能产生显著梯度

     - 优化过程更"宽松"，允许策略做出更大的调整

     - 模型可以更自由地偏离参考模型

3) 数学上的解释：

   - $r_\theta = \log\frac{\pi_\theta}{\pi_\text{ref}}$ 正是KL散度的项

   - β越大，这个比值的变化越受限

   - β越小，这个比值可以有更大的变化空间

4) 直观理解：

   - β可以理解为"保守程度"的调节器

   - 大β = 保守优化 = 小KL散度

   - 小β = 激进优化 = 允许大KL散度

这就是为什么说β参数实际上在调节KL散度的约束强度。

5) DPO可以被视为在最大化奖励的同时，通过$\beta$参数软约束KL散度。这种设计使得模型既能学习偏好，又不会过分偏离参考模型的行为分布。


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
