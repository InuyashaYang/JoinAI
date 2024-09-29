[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

## 0 大模型 Transformer 算法架构

### Transformer 编码器（Encoder）结构


| 层 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 输入嵌入 | 原始输入序列 | 词嵌入 | $X \in \mathbb{R}^{L \times d_{model}}$ |
| 位置编码 | 词嵌入 $X \in \mathbb{R}^{L \times d_{model}}$ | 添加位置信息：$I = X + P$ <br> 其中 $P \in \mathbb{R}^{L \times d_{model}}$ 是位置编码矩阵 | $I \in \mathbb{R}^{L \times d_{model}}$ |
| 多头自注意力 | $I \in \mathbb{R}^{L \times d_{model}}$ | 1. $Q = IW_Q, K = IW_K, V = IW_V$ <br> 2. $A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})$ <br> 3. $Z = AV$ <br> 4. 拼接多头: $Z_{concat} = [Z_1; Z_2; ...; Z_h]$ <br> 5. $\text{MultiHead} = Z_{concat}W_O$ <br> 其中所有中间结果维度与输入输出一致 | $X_{attn} = \text{MultiHead}(Q,K,V) \in \mathbb{R}^{L \times d_{model}}$ |
| 残差连接 + 层归一化 | 多头注意力输出 $X_{attn} \in \mathbb{R}^{L \times d_{model}}$ <br> 原始输入 $I \in \mathbb{R}^{L \times d_{model}}$ | 1. 残差连接：$X_{residual} = X_{attn} + I$ <br><br> 2. 计算均值和标准差：<br> $\mu = \frac{1}{d_{model}}\sum_{j=1}^{d_{model}} X_{residual_{:,j}} \in \mathbb{R}^{L \times 1}$ <br> $\sigma = \sqrt{\frac{1}{d_{model}}\sum_{j=1}^{d_{model}} (X_{residual_{:,j}} - \mu)^2} \in \mathbb{R}^{L \times 1}$ <br><br> 3. 层归一化：<br> $X_{norm} = \gamma \odot (\frac{X_{residual} - \mu \mathbf{1}^T}{(\sigma + \varepsilon)\mathbf{1}^T}) + \beta$ <br><br> 其中：<br> - $\mathbf{1} \in \mathbb{R}^{1 \times d_{model}}$ 是全1向量<br> - $\gamma, \beta \in \mathbb{R}^{1 \times d_{model}}$ 是可学习参数<br> - $\odot$ 表示逐元素乘法（使用广播）<br> - 除法也是逐元素操作（使用广播） | $X_{norm} \in \mathbb{R}^{L \times d_{model}}$ |
| 前馈神经网络 (FFN) | 上一层输出 $X_{norm} \in \mathbb{R}^{L \times d_{model}}$ | $FFN(X_{norm}) = \max(0, X_{norm}W_1 + b_1)W_2 + b_2$ <br> 其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ <br> $b_1 \in \mathbb{R}^{1 \times d_{ff}}$, $b_2 \in \mathbb{R}^{1 \times d_{model}}$ <br> $d_{ff}$ 通常大于 $d_{model}$，例如 $d_{ff} = 4 \times d_{model}$ | $X_{ffn} \in \mathbb{R}^{L \times d_{model}}$ |
| 残差连接 + 层归一化 | FFN输出 $X_{ffn} \in \mathbb{R}^{L \times d_{model}}$ <br> FFN输入 $X_{norm} \in \mathbb{R}^{L \times d_{model}}$ | 1. 残差连接：$X_{residual} = X_{ffn} + X_{norm}$ <br><br> 2. 计算均值和标准差：<br> $\mu = \frac{1}{d_{model}}\sum_{j=1}^{d_{model}} X_{residual_{:,j}} \in \mathbb{R}^{L \times 1}$ <br> $\sigma = \sqrt{\frac{1}{d_{model}}\sum_{j=1}^{d_{model}} (X_{residual_{:,j}} - \mu)^2} \in \mathbb{R}^{L \times 1}$ <br><br> 3. 层归一化：<br> $X_{out} = \gamma \odot (\frac{X_{residual} - \mu \mathbf{1}^T}{(\sigma + \varepsilon)\mathbf{1}^T}) + \beta$ <br><br>  | $X_{out} \in \mathbb{R}^{L \times d_{model}}$ |



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

注意：整个解码器结构以编码器输出 $E \in \mathbb{R}^{L \times d_{model}}$ 作为额外输入，用于交叉注意力层。

| 层 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 输入嵌入 | 目标序列 | 词嵌入 | $Y \in \mathbb{R}^{T \times d_{model}}$ |
| 位置编码 | 词嵌入 $Y$ | 添加位置信息 | $I = Y + P \in \mathbb{R}^{T \times d_{model}}$ |
| 掩码多头自注意力 | $I$ | 1. $Q = IW_Q, K = IW_K, V = IW_V$ <br> 2. $A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + M)$ <br> 3. $Z = AV$ <br> 4. 拼接多头: $Z_{concat} = [Z_1; Z_2; ...; Z_h]$ <br> 5. $\text{MaskedMultiHead} = Z_{concat}W_O$ <br><br> 其中 $M$ 是掩码矩阵，用于防止关注未来位置 | $X_{attn} = \text{MaskedMultiHead}(Q,K,V) \in \mathbb{R}^{T \times d_{model}}$ |
| 残差连接 + 层归一化 | 掩码多头注意力输出 $X_{attn}$ <br> 原始输入 $I$ | 1. 残差连接：$X_{residual} = X_{attn} + I$ <br> 2. 计算均值和标准差：<br> $\mu = \frac{1}{d_{model}}\sum_{j=1}^{d_{model}} X_{residual_{:,j}} \in \mathbb{R}^{T \times 1}$ <br> $\sigma = \sqrt{\frac{1}{d_{model}}\sum_{j=1}^{d_{model}} (X_{residual_{:,j}} - \mu)^2} \in \mathbb{R}^{T \times 1}$ <br> 3. 层归一化：<br> $X_{norm} = \gamma \odot (\frac{X_{residual} - \mu \mathbf{1}^T}{(\sigma + \varepsilon)\mathbf{1}^T}) + \beta$ <br><br> 其中：<br> - $\mathbf{1} \in \mathbb{R}^{1 \times d_{model}}$ 是全1向量<br> - $\gamma, \beta \in \mathbb{R}^{1 \times d_{model}}$ 是可学习参数<br> - $\odot$ 表示逐元素乘法（使用广播）<br> - 除法也是逐元素操作（使用广播） | $X_{norm} \in \mathbb{R}^{T \times d_{model}}$ |
| 多头交叉注意力 | $X_{norm}$, 编码器输出 $E$ | 1. $Q = X_{norm}W_Q$ <br> 2. $K = EW_K$ <br> 3. $V = EW_V$ <br> 4. $A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})$ <br> 5. $Z = AV$ <br> 6. 拼接多头: $Z_{concat} = [Z_1; Z_2; ...; Z_h]$ <br> 7. $\text{CrossMultiHead} = Z_{concat}W_O$ | $X_{cross} = \text{CrossMultiHead}(Q,K,V) \in \mathbb{R}^{T \times d_{model}}$ |
| 残差连接 + 层归一化 | 交叉注意力输出 $X_{cross}$ <br> 上一层输出 $X_{norm}$ | 1. 残差连接：$X_{residual} = X_{cross} + X_{norm}$ <br> 2. 计算均值和标准差：<br> $\mu = \frac{1}{d_{model}}\sum_{j=1}^{d_{model}} X_{residual_{:,j}} \in \mathbb{R}^{T \times 1}$ <br> $\sigma = \sqrt{\frac{1}{d_{model}}\sum_{j=1}^{d_{model}} (X_{residual_{:,j}} - \mu)^2} \in \mathbb{R}^{T \times 1}$ <br> 3. 层归一化：<br> $X_{norm2} = \gamma \odot (\frac{X_{residual} - \mu \mathbf{1}^T}{(\sigma + \varepsilon)\mathbf{1}^T}) + \beta$ | $X_{norm2} \in \mathbb{R}^{T \times d_{model}}$ |
| 前馈神经网络 (FFN) | $X_{norm2}$ | $FFN(X_{norm2}) = \max(0, X_{norm2}W_1 + b_1)W_2 + b_2$ <br> 其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ <br> $b_1 \in \mathbb{R}^{1 \times d_{ff}}$, $b_2 \in \mathbb{R}^{1 \times d_{model}}$ <br> $d_{ff}$ 通常大于 $d_{model}$，例如 $d_{ff} = 4 \times d_{model}$ | $X_{ffn} \in \mathbb{R}^{T \times d_{model}}$ |
| 残差连接 + 层归一化 | FFN输出 $X_{ffn}$ <br> FFN输入 $X_{norm2}$ | 1. 残差连接：$X_{residual} = X_{ffn} + X_{norm2}$ <br> 2. 计算均值和标准差：<br> $\mu = \frac{1}{d_{model}}\sum_{j=1}^{d_{model}} X_{residual_{:,j}} \in \mathbb{R}^{T \times 1}$ <br> $\sigma = \sqrt{\frac{1}{d_{model}}\sum_{j=1}^{d_{model}} (X_{residual_{:,j}} - \mu)^2} \in \mathbb{R}^{T \times 1}$ <br> 3. 层归一化：<br> $X_{out} = \gamma \odot (\frac{X_{residual} - \mu \mathbf{1}^T}{(\sigma + \varepsilon)\mathbf{1}^T}) + \beta$ | $X_{out} \in \mathbb{R}^{T \times d_{model}}$ |
| 线性层(最后一层decoder) | $X_{out}$ | $\text{Linear}(X_{out}) = X_{out}W + b$ <br> 其中 $W \in \mathbb{R}^{d_{model} \times \text{vocab\_size}}$ | $\in \mathbb{R}^{T \times \text{vocab\_size}}$ |
| Softmax(最后一层decoder) | 线性层输出 | $\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$ <br> 对每个时间步的输出进行softmax操作 | $\in \mathbb{R}^{T \times \text{vocab\_size}}$ |

注意：
1. $T$ 是目标序列的长度
2. $L$ 是源序列的长度
3. $d_{model}$ 是模型的隐藏维度
4. 多头注意力中，$d_k = d_{model} / \text{num\_heads}$
5. 层归一化 (LN) 公式：$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$
6. 解码器中的掩码多头自注意力使用了掩码矩阵 $M$，以防止关注未来位置
7. 交叉注意力层使用编码器的输出 $E$ 作为 Key 和 Value

# Transformer中的数据流动

| 层 | 输入 | 输出 | 输出去向 |
|---|---|---|---|
| 输入嵌入 | 源序列 | $X_0 \in \mathbb{R}^{L \times d_{model}}$ | Encoder第1层 |
| Encoder第1层 | $X_0$ | $X_1 \in \mathbb{R}^{L \times d_{model}}$ | Encoder第2层 |
| Encoder第2-5层 | 上一层输出 | $X_i \in \mathbb{R}^{L \times d_{model}}$ | 下一个Encoder层 |
| Encoder第6层 | $X_5$ | $E \in \mathbb{R}^{L \times d_{model}}$ | 所有Decoder层的交叉注意力 |
| 目标嵌入 | 目标序列 | $Y_0 \in \mathbb{R}^{T \times d_{model}}$ | Decoder第1层 |
| Decoder第1层 | $Y_0$, $E$ | $Y_1 \in \mathbb{R}^{T \times d_{model}}$ | Decoder第2层 |
| Decoder第2-5层 | 上一层输出, $E$ | $Y_i \in \mathbb{R}^{T \times d_{model}}$ | 下一个Decoder层 |
| Decoder第6层 | $Y_5$, $E$ | $Y_6 \in \mathbb{R}^{T \times d_{model}}$ | 线性层 |
| 线性层 | $Y_6$ | $Z \in \mathbb{R}^{T \times \text{vocab\_size}}$ | Softmax |
| Softmax | $Z$ | 概率分布 $\in \mathbb{R}^{T \times \text{vocab\_size}}$ | 最终输出 |

注：
- $L$: 源序列长度
- $T$: 目标序列长度
- $d_{model}$: 模型维度
- $\text{vocab\_size}$: 词汇表大小


# 模型更新时更新的参数
## Encoder


| 组件 | 参数 | 维度 | 描述 |
|------|------|------|------|
| 多头自注意力 | $W_Q$ | $\mathbb{R}^{d_{model} \times d_k}$ | 查询权重（每个头） |
| | $W_K$ | $\mathbb{R}^{d_{model} \times d_k}$ | 键权重（每个头） |
| | $W_V$ | $\mathbb{R}^{d_{model} \times d_v}$ | 值权重（每个头） |
| | $W_O$ | $\mathbb{R}^{hd_v \times d_{model}}$ | 输出权重 |
| 层归一化 (注意力后) | $\gamma$ | $\mathbb{R}^{d_{model}}$ | 缩放参数 |
| | $\beta$ | $\mathbb{R}^{d_{model}}$ | 偏移参数 |
| 前馈神经网络 | $W_1$ | $\mathbb{R}^{d_{model} \times d_{ff}}$ | 第一层权重 |
| | $b_1$ | $\mathbb{R}^{d_{ff}}$ | 第一层偏置 |
| | $W_2$ | $\mathbb{R}^{d_{ff} \times d_{model}}$ | 第二层权重 |
| | $b_2$ | $\mathbb{R}^{d_{model}}$ | 第二层偏置 |
| 层归一化 (FFN后) | $\gamma$ | $\mathbb{R}^{d_{model}}$ | 缩放参数 |
| | $\beta$ | $\mathbb{R}^{d_{model}}$ | 偏移参数 |

注意：
- $L$ 是输入序列的长度
- $h$ 是注意力头的数量
- $d_k$ 和 $d_v$ 通常等于 $d_{model} / h$
- $d_{ff}$ 是前馈网络的内部维度，通常大于 $d_{model}$
- 除了输入嵌入，其他参数在每个编码器层中都会重复出现
- 如果编码器有多层（例如标准 Transformer 中的 6 层），这些参数会在每层中重复


## Decoder

| 组件 | 参数 | 维度 | 描述 |
|------|------|------|------|
| 掩码多头自注意力 | $W_Q$ | $\mathbb{R}^{d_{model} \times d_k}$ | 查询权重（每个头） |
| | $W_K$ | $\mathbb{R}^{d_{model} \times d_k}$ | 键权重（每个头） |
| | $W_V$ | $\mathbb{R}^{d_{model} \times d_v}$ | 值权重（每个头） |
| | $W_O$ | $\mathbb{R}^{hd_v \times d_{model}}$ | 输出权重 |
| 多头交叉注意力 | $W_Q$ | $\mathbb{R}^{d_{model} \times d_k}$ | 查询权重（每个头） |
| | $W_K$ | $\mathbb{R}^{d_{model} \times d_k}$ | 键权重（每个头） |
| | $W_V$ | $\mathbb{R}^{d_{model} \times d_v}$ | 值权重（每个头） |
| | $W_O$ | $\mathbb{R}^{hd_v \times d_{model}}$ | 输出权重 |
| 前馈神经网络 | $W_1$ | $\mathbb{R}^{d_{model} \times d_{ff}}$ | 第一层权重 |
| | $b_1$ | $\mathbb{R}^{d_{ff}}$ | 第一层偏置 |
| | $W_2$ | $\mathbb{R}^{d_{ff} \times d_{model}}$ | 第二层权重 |
| | $b_2$ | $\mathbb{R}^{d_{model}}$ | 第二层偏置 |
| 层归一化 | $\gamma$ | $\mathbb{R}^{d_{model}}$ | 缩放参数 |
| | $\beta$ | $\mathbb{R}^{d_{model}}$ | 偏移参数 |
| 输出线性层 | $W_{out}$ | $\mathbb{R}^{d_{model} \times \text{vocab\_size}}$ | 输出权重 |
| | $b_{out}$ | $\mathbb{R}^{\text{vocab\_size}}$ | 输出偏置 |

注意：
- $h$ 是注意力头的数量
- $d_k$ 和 $d_v$ 通常等于 $d_{model} / h$
- $d_{ff}$ 是前馈网络的内部维度，通常大于 $d_{model}$
- 除了输入嵌入和最终输出层，其他参数在每个解码器层中都会重复出现
- 如果解码器有多层（例如标准 Transformer 中的 6 层），这些参数会在每层中重复

# 一些问题

## 为什么在编码器和解码器层中有多次残差连接+层归一化？

### 编码器层

1. **第一次残差连接+层归一化（在自注意力之后）**
   - **保持信息流动**：允许原始信息直接传递，有助于解决深层网络中的梯度消失问题。
   - **稳定训练**：层归一化有助于稳定深层网络的训练过程。
   - **增强特征**：结合了自注意力机制的上下文信息和原始输入信息。

2. **第二次残差连接+层归一化（在前馈网络之后）**
   - **非线性转换**：在保留原始信息的同时，允许模型进行复杂的非线性变换。
   - **深度特征提取**：前馈网络可以提取更深层次的特征，残差连接确保这些特征与原始信息结合。
   - **梯度流动**：再次缓解梯度消失问题，使得即使在很深的网络中，信息也能有效地向后传播。

### 解码器层

1. **第一次残差连接+层归一化（在掩码自注意力之后）**
   - **保持目标序列信息**：确保模型在处理目标序列时不会丢失原始输入信息。
   - **自回归特性**：在自回归生成过程中，有助于保持已生成部分的连贯性。

2. **第二次残差连接+层归一化（在交叉注意力之后）**
   - **融合编码器信息**：允许解码器有效地整合来自编码器的信息，同时保留自身的表示。
   - **平衡源和目标信息**：帮助模型在源语言（编码器输出）和目标语言（解码器状态）之间取得平衡。

3. **第三次残差连接+层归一化（在前馈网络之后）**
   - **功能类似编码器**：进行深度特征提取和非线性变换。
   - **整合多源信息**：结合了自注意力、交叉注意力和前馈网络的输出，形成丰富的表示。

## 总结

- **信息流动**：多次残差连接确保了原始信息和转换后的信息能够在深层网络中有效传播。
- **梯度流动**：有助于解决深度学习中的梯度消失问题，使得模型更容易训练。
- **特征融合**：每一步都融合了不同层次的特征，从原始输入到高度抽象的表示。
- **稳定性**：层归一化在每一步后都能稳定激活值，有助于训练的稳定性和收敛速度。
- **灵活性**：这种结构允许模型在保留必要信息的同时，进行复杂的非线性变换，增强了模型的表达能力。


## 前馈神经网络（FFN）在 Transformer 中的作用

前馈神经网络是 Transformer 架构中的一个关键组件，在编码器和解码器的每一层都有使用。让我们详细探讨它的作用：

### 编码器中的 FFN

1. **非线性变换**
   - 引入非线性：通过激活函数（如 ReLU）引入非线性，增强模型的表达能力。
   - 复杂特征提取：能够学习和提取更复杂的特征表示。

2. **增加模型容量**
   - 参数增加：FFN 通常有较大的隐藏层（如 4 倍的 $d_{model}$），大幅增加模型参数数量。
   - 学习能力提升：更多参数意味着模型可以学习更复杂的模式和关系。

3. **位置特定处理**
   - 独立转换：对每个位置的表示进行独立的转换，补充了自注意力的全局处理。
   - 局部特征增强：可以捕捉和强化特定位置的特征。

4. **信息整合**
   - 综合处理：在自注意力机制之后，进一步处理和整合信息。
   - 特征重组：重新组合和调整通过注意力机制获得的特征。

### 解码器中的 FFN

1. **功能类似编码器**
   - 在解码器中，FFN 的基本功能与编码器中相同。

2. **多源信息处理**
   - 整合多方面信息：在解码器中，FFN 处理的是经过自注意力和交叉注意力后的信息。
   - 复杂决策：帮助模型在生成输出时做出更复杂的决策。

3. **输出准备**
   - 特征调整：为最终的输出层（通常是一个线性层加 softmax）准备合适的特征表示。

### 共同特点

1. **维度变换**
   - 扩展后收缩：通常先将维度从 $d_{model}$ 扩展到 $d_{ff}$（如 4 * $d_{model}$），然后再压缩回 $d_{model}$。
   - 结构：$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$

2. **计算效率**
   - 并行处理：FFN 可以对序列中的每个位置并行计算，提高效率。

3. **残差连接**
   - 与残差连接结合：FFN 的输出通过残差连接与输入相加，有助于信息和梯度的流动。

### 总结

前馈神经网络在 Transformer 中扮演着重要角色，它通过增加非线性、提升模型容量、进行位置特定处理来增强模型的表达能力。FFN 与注意力机制相辅相成，共同构建了 Transformer 强大的学习和推理能力。在编码器和解码器中，FFN 帮助模型更好地理解输入序列和生成高质量的输出序列。

[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)
