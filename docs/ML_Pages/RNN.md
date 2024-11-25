
## 循环神经网络 (RNN)

核心结构：**隐状态 → 输入 → 更新 → 输出 → 重复**

### 1. 数学表示

对时间步 $t$：$\mathbf{h}_t = \sigma(\mathbf{W}_h\mathbf{h}_{t-1} + \mathbf{W}_x\mathbf{x}_t + \mathbf{b})$

- $\mathbf{h}_t$: 当前隐状态
- $\mathbf{x}_t$: 当前输入
- $\mathbf{W}_h, \mathbf{W}_x$: 权重矩阵
- $\mathbf{b}$: 偏置
- $\sigma(\cdot)$: 激活函数

### 2. 网络结构

| 组件 | 公式 |
|------|------|
| 输入 | $\mathbf{x}_t \in \mathbb{R}^d$ |
| 隐状态更新 | $\mathbf{h}_t = \sigma(\mathbf{W}_h\mathbf{h}_{t-1} + \mathbf{W}_x\mathbf{x}_t + \mathbf{b})$ |
| 输出 | $\mathbf{y}_t = \mathbf{W}_o\mathbf{h}_t + \mathbf{b}_o$ |


### 3. RNN 训练方法

#### 3.1 反向传播通过时间 (BPTT)

核心思想：展开 RNN 为等效的前馈网络，然后应用标准反向传播

数学表示：
- 损失函数：$L = \sum_{t=1}^T L_t(\mathbf{y}_t, \mathbf{\hat{y}}_t)$
- 梯度计算：$\frac{\partial L}{\partial \mathbf{W}} = \sum_{t=1}^T \frac{\partial L_t}{\partial \mathbf{W}}$

#### 3.2 截断 BPTT

目的：解决长序列梯度消失/爆炸问题

方法：限制反向传播的时间步数 $k$
$$\frac{\partial L}{\partial \mathbf{W}} \approx \sum_{t=T-k+1}^T \frac{\partial L_t}{\partial \mathbf{W}}$$

#### 3.3 优化技巧

| 技巧 | 描述 | 公式/方法 |
|------|------|-----------|
| 梯度裁剪 | 防止梯度爆炸 | $\mathbf{g} \leftarrow \min(1, \frac{\theta}{\|\|\mathbf{g}\|\|}) \mathbf{g}$ |
| 长短期记忆 (LSTM) | 改进的 RNN 结构 | 使用门控机制 |
| 梯度累积 | 处理超长序列 | 多批次累积梯度后更新 |

#### 3.4 正则化方法

- Dropout：适用于非循环连接
- 循环 Dropout：在时间步之间保持一致的 dropout mask
- L2 正则化：权重衰减



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
