[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

## 1. 马尔可夫决策过程 (MDP)

### 1.1 定义

MDP 是一个五元组 $(S, A, P, R, \gamma)$，其中：

| 符号 | 含义 | 说明 |
|------|------|------|
| $S$ | 状态空间 | 有限或无限 |
| $A$ | 动作空间 | 有限或无限 |
| $P$ | 状态转移概率 | $P: S \times A \times S \rightarrow [0,1]$ |
| $R$ | 奖励函数 | $R: S \times A \times S \rightarrow \mathbb{R}$ |
| $\gamma$ | 折扣因子 | $\gamma \in [0,1]$ |

### 1.2 矩阵表示

对于有限状态和动作空间，可以用矩阵表示：

- 转移概率矩阵 $\mathbf{P}_a$：对每个动作 $a$，$[\mathbf{P}_a]_{ij} = P(s_j|s_i,a)$
- 奖励矩阵 $\mathbf{R}_a$：$[\mathbf{R}_a]_{ij} = R(s_i,a,s_j)$

## 2. 价值函数

### 2.1 定义

| 函数 | 定义 | 说明 |
|------|------|------|
| 状态价值函数 | $V^\pi(s) = \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \| S_0 = s]$ | 从状态 $s$ 开始的期望累积奖励 |
| 动作价值函数 | $Q^\pi(s,a) = \mathbb{E}_\pi[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \| S_0 = s, A_0 = a]$ | 从状态 $s$ 执行动作 $a$ 开始的期望累积奖励 |

### 2.2 矩阵表示

对于有限 MDP，可以将价值函数表示为向量：

- 状态价值向量：$\mathbf{V}^\pi = [V^\pi(s_1), \ldots, V^\pi(s_n)]^T$
- 动作价值矩阵：$\mathbf{Q}^\pi = [Q^\pi(s_i,a_j)]_{i,j}$

## 3. 贝尔曼方程

### 3.1 状态价值函数的贝尔曼方程

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

紧凑形式：$V^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi V^\pi$

其中，$\mathbf{R}^\pi$ 和 $\mathbf{P}^\pi$ 是策略 $\pi$ 下的期望奖励向量和转移概率矩阵。

### 3.2 动作价值函数的贝尔曼方程

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

矩阵形式：$\mathbf{Q}^\pi = \mathbf{R} + \gamma \mathbf{P} (\mathbf{\Pi} \mathbf{Q}^\pi)$

其中，$\mathbf{\Pi}$ 是策略矩阵，$[\mathbf{\Pi}]_{ij} = \pi(a_j|s_i)$。

### 3.3 最优贝尔曼方程

状态价值：$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$

动作价值：$Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$

矩阵形式：
- $\mathbf{V}^* = \max_a (\mathbf{R}_a + \gamma \mathbf{P}_a \mathbf{V}^*)$
- $\mathbf{Q}^* = \mathbf{R} + \gamma \mathbf{P} \max_a \mathbf{Q}^*$

4. 最优价值函数:

$V^*(s) = \max_\pi V^\pi(s)$
$Q^*(s,a) = \max_\pi Q^\pi(s,a)$

5. 策略改进定理:

如果对所有 $s \in S$，有 $Q^\pi(s,\pi'(s)) \geq V^\pi(s)$，则 $V^{\pi'}(s) \geq V^\pi(s)$。

6. 策略迭代:

策略评估：$V_{k+1}^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k^\pi(s')]$

策略改进：$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$

7. 值迭代:

$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$

8. Q-学习:

$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]$

9. SARSA:

$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma Q(s',a') - Q(s,a)]$

10. 策略梯度定理:

$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$

11. Actor-Critic 方法:

Actor 更新：$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) A(s,a)$
Critic 更新：$w \leftarrow w - \beta \nabla_w (R + \gamma V_w(s') - V_w(s))^2$

其中 $A(s,a) = Q(s,a) - V(s)$ 是优势函数。

12. 探索与利用:

$\epsilon$-贪心策略：以 $1-\epsilon$ 的概率选择最优动作，以 $\epsilon$ 的概率随机探索。

Boltzmann 探索：$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$

13. 函数近似:

使用参数化函数 $Q_\theta(s,a)$ 或 $V_\theta(s)$ 来近似价值函数。

14. 经验回放:

从经验池 $D$ 中采样 $(s,a,r,s')$ 进行学习，减少样本相关性。

15. 目标网络:

使用单独的目标网络 $Q_{\theta'}$ 来计算目标值，提高训练稳定性：

$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}[(r + \gamma \max_{a'} Q_{\theta'}(s',a') - Q_\theta(s,a))^2]$

16. 双 Q-学习:

使用两个 Q 网络来减少过估计偏差：

$y = r + \gamma Q_{\theta_2}(s', \arg\max_{a'} Q_{\theta_1}(s',a'))$


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
