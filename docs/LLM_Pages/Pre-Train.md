[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

# 预训练阶段

# Transformer 模型训练逻辑 (Teacher Forcing 视角)

## 准备阶段

1. **输入准备**
   - 源序列: $X = [x_1, x_2, ..., x_n]$
   - 目标序列: $Y = [y_1, y_2, ..., y_m]$
   - 添加特殊标记: 
     * 源序列: $X' = [<START>, x_1, x_2, ..., x_n, <END>]$
     * 目标序列: $Y' = [<START>, y_1, y_2, ..., y_m, <END>]$

2. **训练数据构造**
   - 解码器输入: $Y_{in} = [<START>, y_1, y_2, ..., y_m]$
   - 解码器目标: $Y_{out} = [y_1, y_2, ..., y_m, <END>]$

## 前向传播

### 编码器处理
```
Encoder_Output = Encoder(X')
```

### 解码器处理
```
for t in range(1, m+1):
    Decoder_Input = Y_{in}[:t]  # [<START>, y_1, ..., y_{t-1}]
    Decoder_Output = Decoder(Decoder_Input, Encoder_Output)
    Prediction[t] = Linear(Decoder_Output[-1])  # 只使用最后一个时间步的输出
```

## 损失计算

```
Loss = 0
for t in range(1, m+1):
    Loss += CrossEntropyLoss(Prediction[t], Y_{out}[t])
```

## 反向传播和参数更新

```
Loss.backward()
Optimizer.step()
```

## 关键点说明

1. **并行计算**
   - 尽管上述逻辑是按时间步描述的，Transformer 实际上可以并行处理整个序列。

2. **掩码自注意力**
   - 在解码器中，使用掩码确保位置 t 只能注意到 1 到 t-1 的位置。

3. **Teacher Forcing**
   - 在训练时，解码器始终使用正确的前缀 ($Y_{in}[:t]$) 来预测下一个词，而不是使用自己的预测。

4. **预测和目标错位**
   - 预测 $Prediction[t]$ 对应的目标是 $Y_{out}[t]$，即下一个词。

5. **损失计算**
   - 对每个时间步的预测单独计算损失，然后累加。

6. **梯度累积**
   - 反向传播时，梯度会通过所有时间步累积，允许模型学习长期依赖。


# 交叉熵损失计算

对于时间步 $t$，我们计算 `CrossEntropyLoss(Prediction[t], Y_{out}[t])` 如下：

## 输入

- $Prediction[t] \in \mathbb{R}^{\text{vocab\_size}}$：模型在时间步 $t$ 的输出（logits）
- $Y_{out}[t] \in \{1, 2, ..., \text{vocab\_size}\}$：时间步 $t$ 的真实标签（一个整数，表示正确单词的索引）

## 步骤

1. **应用 Softmax 函数**

   将 logits 转换为概率分布：

   $$P[t] = \text{softmax}(Prediction[t])$$

   其中，对于每个类别 $i$：

   $$P[t]_i = \frac{\exp(Prediction[t]_i)}{\sum_{j=1}^{\text{vocab\_size}} \exp(Prediction[t]_j)}$$

2. **计算交叉熵**

   $$\text{CrossEntropyLoss}(Prediction[t], Y_{out}[t]) = -\log(P[t]_{Y_{out}[t]})$$

   这里，$P[t]_{Y_{out}[t]}$ 是正确类别的预测概率。

## 数值稳定性考虑

在实际实现中，为了数值稳定性，通常会结合 softmax 和对数操作：

$$\text{CrossEntropyLoss}(Prediction[t], Y_{out}[t]) = -Prediction[t]_{Y_{out}[t]} + \log\left(\sum_{j=1}^{\text{vocab\_size}} \exp(Prediction[t]_j)\right)$$

## 示例

假设：
- $\text{vocab\_size} = 5$
- $Prediction[t] = [2.0, 1.0, 0.1, 3.0, -1.0]$
- $Y_{out}[t] = 4$ （正确答案是第4个单词）

计算过程：

1. 应用 softmax：
   $$P[t] \approx [0.244, 0.090, 0.037, 0.665, 0.012]$$

2. 计算损失：
   $$\text{CrossEntropyLoss} = -\log(0.665) \approx 0.408$$

这个损失值反映了模型预测的准确程度。损失越小，表示模型的预测越接近真实标签。

在实际训练中，我们会对所有时间步的损失求和或求平均，得到整个序列的总体损失，然后用这个总体损失来更新模型参数。


[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)