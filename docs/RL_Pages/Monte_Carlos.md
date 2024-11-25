
# 蒙特卡洛树搜索(MCTS)数学原理

## 1. 基本结构

| 组件 | 数学表示 |
|------|----------|
| 决策树 | $T = (V, E)$ |
| 节点集 | $V$ |
| 边集 | $E$ |

## 2. 节点表示

每个节点 $v \in V$ 包含：

- 状态 $s$
- 访问次数 $n(v)$
- 累积价值 $Q(v)$

## 3. UCB1公式

$$UCB1(v) = \frac{Q(v)}{n(v)} + C\sqrt{\frac{\ln N(v)}{n(v)}}$$

其中：
- $C$ 是探索参数
- $N(v)$ 是父节点的访问次数

## 4. MCTS四个主要步骤

### a) 选择(Selection)
---

从根节点开始，递归地选择子节点，直到达到叶节点：

$$v_{selected} = \arg\max_{v \in children(v)} UCB1(v)$$

### b) 扩展(Expansion)
---

如果叶节点不是终止状态，则添加一个或多个子节点。

### c) 模拟(Simulation)
---

从新添加的节点开始，使用随机策略进行游戏模拟直到终止状态，得到奖励 $r \in \mathbb{R}$。

### d) 反向传播(Backpropagation)
---

更新路径上所有节点的统计信息：

| 更新项 | 公式 |
|--------|------|
| 访问次数 | $n(v) = n(v) + 1$ |
| 累积价值 | $Q(v) = Q(v) + r$ |

## 5. 最优动作选择

$$a^* = \arg\max_{a \in A(s)} n(v_a)$$

其中 $A(s)$ 是当前状态 $s$ 的可用动作集。

## 6. 收敛性

$$\lim_{t \to \infty} P(a_t = a^*) = 1$$

其中 $a_t$ 是第 $t$ 次迭代后选择的动作。



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
