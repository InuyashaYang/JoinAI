[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

## 0 大模型 Transformer 算法架构

### Transformer 编码器（Encoder）结构

| 层 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 输入嵌入 | 原始输入序列 | 词嵌入 | $X \in \mathbb{R}^{L \times d_{model}}$ |
| 位置编码 | 词嵌入 $X$ | 添加位置信息 | $I = X + P \in \mathbb{R}^{L \times d_{model}}$ |
| 多头自注意力 | $I$ | 1. $Q = IW_Q, K = IW_K, V = IW_V$ <br> 2. $A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})$ <br> 3. $Z = AV$ <br> 4. 拼接多头: $Z_{concat} = [Z_1; Z_2; ...; Z_h]$ <br> 5. $\text{MultiHead} = Z_{concat}W_O$ | $\text{MultiHead}(Q,K,V) \in \mathbb{R}^{L \times d_{model}}$ |
| 残差连接 + 层归一化 | 多头注意力输出 + I | 1. 残差连接: 将多头注意力的输出与原始输入相加<br>2. 层归一化: 对结果进行归一化处理<br>3. 数学表达: LN(x) = γ ⊙ (x - μ) / (σ + ε) + β<br>   其中 μ 是均值，σ 是标准差，γ 和 β 是可学习的参数 | ∈ ℝ^(L×d_model) |
| 前馈神经网络 (FFN) | 上一层输出 | 1. 第一层线性变换: xW_1 + b_1<br>2. ReLU激活函数: max(0, ·)<br>3. 第二层线性变换: (·)W_2 + b_2<br>4. W_1 ∈ ℝ^(d_model×d_ff), W_2 ∈ ℝ^(d_ff×d_model)<br>5. d_ff 通常大于 d_model，例如 d_ff = 4 * d_model<br>6. FFN(x) = max(0, xW_1 + b_1)W_2 + b_2 | ∈ ℝ^(L×d_model) |
| 残差连接 + 层归一化 | FFN输出 + 上一层输入 | 1. 残差连接: 将FFN的输出与其输入相加<br>2. 层归一化: 对结果进行归一化处理<br>3. 数学表达: LN(x) = γ ⊙ (x - μ) / (σ + ε) + β<br>   其中 μ 是均值，σ 是标准差，γ 和 β 是可学习的参数<br>4. 这一步有助于稳定深度网络的训练过程 | ∈ ℝ^(L×d_model) |


注意：

1. $L$ 是输入序列的长度

2. $d_{model}$ 是模型的隐藏维度

3. 多头注意力中，$d_k = d_{model} / \text{num\_heads}$

4. 层归一化 (LN) 公式：$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$

<details> <summary>⊙ Hadamard乘积（元素级乘法）解释</summary>
Hadamard乘积，也称为元素级乘法（element-wise multiplication），是两个相同维度的矩阵或向量之间的运算。

定义：
对于两个相同维度的矩阵 A 和 B，它们的Hadamard乘积 C = A ⊙ B 定义为：
C[i,j] = A[i,j] * B[i,j]

其中 [i,j] 表示矩阵的第i行第j列元素。

例子：
假设有两个 2x2 矩阵：

A = [1 2]
[3 4]

B = [5 6]
[7 8]

它们的Hadamard乘积为：

C = A ⊙ B = [15 26]
[37 48]
= [5 12]
[21 32]

特点：

结果矩阵与原矩阵维度相同
每个位置的元素是原矩阵对应位置元素的乘积
交换律成立：A ⊙ B = B ⊙ A
与普通矩阵乘法不同
在深度学习中的应用：

在注意力机制中进行缩放
在某些激活函数的计算中
在梯度计算和反向传播中
编程实现：
在numpy中可以直接用 * 操作符：C = A * B
在PyTorch中可以用 torch.mul(A, B) 或 A * B

</details>

### Transformer 解码器（Decoder）结构

| 层 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 输入嵌入 | 目标序列 | 词嵌入 | $Y \in \mathbb{R}^{T \times d_{model}}$ |
| 位置编码 | 词嵌入 $Y$ | 添加位置信息 | $I = Y + P \in \mathbb{R}^{T \times d_{model}}$ |
| 掩码多头自注意力 | $I$ | 1. $Q = IW_Q, K = IW_K, V = IW_V$ <br> 2. $A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + M)$ <br> 3. $Z = AV$ <br> 4. 拼接多头: $Z_{concat} = [Z_1; Z_2; ...; Z_h]$ <br> 5. $\text{MaskedMultiHead} = Z_{concat}W_O$ <br><br> 其中 $M$ 是掩码矩阵，用于防止关注未来位置 | $\text{MaskedMultiHead}(Q,K,V) \in \mathbb{R}^{T \times d_{model}}$ |
| 残差连接 + 层归一化 | 掩码多头注意力输出 + I | 1. 残差连接: 将掩码多头注意力的输出与原始输入相加 <br> 2. 层归一化: 对结果进行归一化处理 <br> 3. 数学表达: $\text{LN}(I + \text{MaskedMultiHead})$ <br> 4. $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$ <br> 其中 $\mu$ 是均值，$\sigma$ 是标准差，$\gamma$ 和 $\beta$ 是可学习的参数 | $\in \mathbb{R}^{T \times d_{model}}$ |
| 多头交叉注意力 | 上一层输出, 编码器输出 | 1. $Q = \text{prev\_output}W_Q$ <br> 2. $K = \text{encoder\_output}W_K$ <br> 3. $V = \text{encoder\_output}W_V$ <br> 4. $A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})$ <br> 5. $Z = AV$ <br> 6. 拼接多头: $Z_{concat} = [Z_1; Z_2; ...; Z_h]$ <br> 7. $\text{CrossMultiHead} = Z_{concat}W_O$ <br><br> 这里的 $K$ 和 $V$ 来自编码器输出 | $\text{CrossMultiHead}(Q,K,V) \in \mathbb{R}^{T \times d_{model}}$ |
| 残差连接 + 层归一化 | 交叉注意力输出 + 上一层输出 | 1. 残差连接: 将交叉注意力的输出与上一层输出相加 <br> 2. 层归一化: 对结果进行归一化处理 <br> 3. 数学表达: $\text{LN}(\text{prev\_output} + \text{CrossMultiHead})$ <br> 4. $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$ <br> 其中 $\mu$ 是均值，$\sigma$ 是标准差，$\gamma$ 和 $\beta$ 是可学习的参数 | $\in \mathbb{R}^{T \times d_{model}}$ |
| 前馈神经网络 (FFN) | 上一层输出 | 1. 第一层线性变换: $xW_1 + b_1$ <br> 2. ReLU激活函数: $\max(0, \cdot)$ <br> 3. 第二层线性变换: $(\cdot)W_2 + b_2$ <br> 4. $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}, W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ <br> 5. $d_{ff}$ 通常大于 $d_{model}$，例如 $d_{ff} = 4 * d_{model}$ <br> 6. $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ | $\in \mathbb{R}^{T \times d_{model}}$ |
| 残差连接 + 层归一化 | FFN输出 + 上一层输入 | 1. 残差连接: 将FFN的输出与其输入相加 <br> 2. 层归一化: 对结果进行归一化处理 <br> 3. 数学表达: $\text{LN}(x + \text{FFN}(x))$ <br> 4. $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$ <br> 其中 $\mu$ 是均值，$\sigma$ 是标准差，$\gamma$ 和 $\beta$ 是可学习的参数 <br> 5. 这一步有助于稳定深度网络的训练过程 | $\in \mathbb{R}^{T \times d_{model}}$ |
| 线性层 | 上一层输出 | $\text{Linear}(x) = xW + b$ <br> 其中 $W \in \mathbb{R}^{d_{model} \times \text{vocab\_size}}$ | $\in \mathbb{R}^{T \times \text{vocab\_size}}$ |
| Softmax | 线性层输出 | $\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$ <br> 对每个时间步的输出进行softmax操作 | $\in \mathbb{R}^{T \times \text{vocab\_size}}$ |

注意：

1. $T$ 是目标序列的长度
2. $d_{model}$ 是模型的隐藏维度
3. 多头注意力中，$d_k = d_{model} / \text{num\_heads}$
4. 层归一化 (LN) 公式：$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$
5. 解码器中的掩码多头自注意力使用了掩码矩阵 $M$，以防止关注未来位置
6. 交叉注意力层使用编码器的输出作为 Key 和 Value


[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)
