[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

## 多层感知器 (MLP)

核心结构：**隐藏层 → 偏置 → 激活 → 重复**

### 1. 数学表示

对第 $l$ 层：$\mathbf{h}_l = \sigma(\mathbf{W}_l\mathbf{h}_{l-1} + \mathbf{b}_l)$

- $\mathbf{h}_l$: 输出
- $\mathbf{W}_l$: 权重
- $\mathbf{b}_l$: 偏置
- $\sigma(\cdot)$: 激活函数

### 2. 网络结构

| 层 | 公式 |
|----|------|
| 输入 | $\mathbf{x} \in \mathbb{R}^{d_0}$ |
| 隐藏 | $\mathbf{h}_l = \sigma(\mathbf{W}_l\mathbf{h}_{l-1} + \mathbf{b}_l)$, $l = 1,\ldots,L-1$ |
| 输出 | $\mathbf{y} = \mathbf{W}_L\mathbf{h}_{L-1} + \mathbf{b}_L$ |

实际上，我们可以把他看作一组函数复合：$f(\mathbf{x}) = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$

其中 $f_l(\mathbf{x}) = \sigma(\mathbf{W}_l\mathbf{x} + \mathbf{b}_l)$

### 3. 关键特性

- 深度：多层结构
- 非线性：激活函数引入
- 全连接：层间全连接

### 4. 复杂度

- 时间（前向传播）：$O(\sum_{l=1}^L d_l d_{l-1})$
- 空间：$O(\sum_{l=1}^L d_l d_{l-1})$

### 5. 核心组件作用

| 组件 | 作用 | 说明 |
|------|------|------|
| 隐藏层 | 特征提取 | 逐层学习，实现复杂模式识别 |
| 激活函数 | 1. 引入非线性<br>2. 防止梯度消失 | 1. 使网络能学习非线性关系<br>2. 如 ReLU，保持梯度流动 |
| 偏置 | 调整阈值 | 增加模型灵活性和适应性 |


### 6. 优缺点

| 优点 | 缺点 |
|------|------|
| 结构简单 | 高维数据效率低 |
| 非线性学习 | 易过拟合 |
| 通用性强 | 难捕捉序列/空间依赖 |
