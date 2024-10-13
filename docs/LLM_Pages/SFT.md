[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

# SFT (Supervised Fine-Tuning) 的数学基础
## 1. $P^*$-Tuning

P 系列微调包括三种主要方法：Prefix-Tuning、Prompt Tuning 和 P-Tuning。这些方法都旨在通过添加或优化输入序列的一部分来适应特定任务，同时保持预训练模型的大部分参数不变。

### 1. Prefix-Tuning


**基本原理：** Prefix-Tuning 在每一层添加可学习的前缀，并确保 Q、K、V 矩阵维度一致。

假设我们有以下参数：
- $d$: 模型的隐藏状态维度
- $n$: 输入序列长度
- $m$: 前缀长度
- $h$: MLP 中间层维度


### 1. Encoder-Decoder 模型 - Encoder 部分

| 步骤 | 描述 | 数学表达式和维度 |
|------|------|------------------|
| 输入扩展 | 添加前缀到输入 | $X_e \in \mathbb{R}^{n \times d}$ <br> $P_{\text{input},e} \in \mathbb{R}^{m \times d}$ <br> $X'_e = [P_{\text{input},e}; X_e] \in \mathbb{R}^{(m+n) \times d}$ |
| Q 计算 | 计算查询矩阵 | $W_Q \in \mathbb{R}^{d \times d}$ <br> $Q_e = X'_eW_Q \in \mathbb{R}^{(m+n) \times d}$ |
| K, V 前缀 | 生成并应用 K, V 前缀 | $W_K, W_V \in \mathbb{R}^{d \times d}$ <br> $P_{K,e}, P_{V,e} \in \mathbb{R}^{m \times d}$ <br> $K_e = [P_{K,e}; X_eW_K] \in \mathbb{R}^{(m+n) \times d}$ <br> $V_e = [P_{V,e}; X_eW_V] \in \mathbb{R}^{(m+n) \times d}$ |
| MLP 生成 | 使用 MLP 生成 K, V 前缀 | $P_{\text{low},e} \in \mathbb{R}^{m \times (d/2)}$ <br> $W_{1,e} \in \mathbb{R}^{h \times (d/2)}, b_{1,e} \in \mathbb{R}^h$ <br> $W_{2,e} \in \mathbb{R}^{2d \times h}, b_{2,e} \in \mathbb{R}^{2d}$ <br> $[P_{K,e}; P_{V,e}] = W_{2,e} \cdot \text{ReLU}(W_{1,e} P_{\text{low},e} + b_{1,e}) + b_{2,e}$ |

### 2. Encoder-Decoder 模型 - Decoder 部分

| 步骤 | 描述 | 数学表达式和维度 |
|------|------|------------------|
| 输入扩展 | 添加前缀到输入 | $Y_d \in \mathbb{R}^{n \times d}$ <br> $P_{\text{input},d} \in \mathbb{R}^{m \times d}$ <br> $Y'_d = [P_{\text{input},d}; Y_d] \in \mathbb{R}^{(m+n) \times d}$ |
| Q 计算 | 计算查询矩阵 | $W_Q \in \mathbb{R}^{d \times d}$ <br> $Q_d = Y'_dW_Q \in \mathbb{R}^{(m+n) \times d}$ |
| K, V 前缀 | 生成并应用 K, V 前缀 | $W_K, W_V \in \mathbb{R}^{d \times d}$ <br> $P_{K,d}, P_{V,d} \in \mathbb{R}^{m \times d}$ <br> $K_d = [P_{K,d}; Y_dW_K] \in \mathbb{R}^{(m+n) \times d}$ <br> $V_d = [P_{V,d}; Y_dW_V] \in \mathbb{R}^{(m+n) \times d}$ |
| MLP 生成 | 使用 MLP 生成 K, V 前缀 | $P_{\text{low},d} \in \mathbb{R}^{m \times (d/2)}$ <br> $W_{1,d} \in \mathbb{R}^{h \times (d/2)}, b_{1,d} \in \mathbb{R}^h$ <br> $W_{2,d} \in \mathbb{R}^{2d \times h}, b_{2,d} \in \mathbb{R}^{2d}$ <br> $[P_{K,d}; P_{V,d}] = W_{2,d} \cdot \text{ReLU}(W_{1,d} P_{\text{low},d} + b_{1,d}) + b_{2,d}$ |

### 3. AutoRegressive 模型

| 步骤 | 描述 | 数学表达式和维度 |
|------|------|------------------|
| 输入扩展 | 添加前缀到输入 | $X \in \mathbb{R}^{n \times d}$ <br> $P_{\text{input}} \in \mathbb{R}^{m \times d}$ <br> $X' = [P_{\text{input}}; X] \in \mathbb{R}^{(m+n) \times d}$ |
| Q 计算 | 计算查询矩阵 | $W_Q \in \mathbb{R}^{d \times d}$ <br> $Q = X'W_Q \in \mathbb{R}^{(m+n) \times d}$ |
| K, V 前缀 | 生成并应用 K, V 前缀 | $W_K, W_V \in \mathbb{R}^{d \times d}$ <br> $P_K, P_V \in \mathbb{R}^{m \times d}$ <br> $K = [P_K; XW_K] \in \mathbb{R}^{(m+n) \times d}$ <br> $V = [P_V; XW_V] \in \mathbb{R}^{(m+n) \times d}$ |
| MLP 生成 | 使用 MLP 生成 K, V 前缀 | $P_{\text{low}} \in \mathbb{R}^{m \times (d/2)}$ <br> $W_1 \in \mathbb{R}^{h \times (d/2)}, b_1 \in \mathbb{R}^h$ <br> $W_2 \in \mathbb{R}^{2d \times h}, b_2 \in \mathbb{R}^{2d}$ <br> $[P_K; P_V] = W_2 \cdot \text{ReLU}(W_1 P_{\text{low}} + b_1) + b_2$ |



注意事项：

1. 所有模型类型中，$Q$、$K$、$V$ 的最终维度都是 $\mathbb{R}^{(m+n) \times d}$，确保了注意力机制的兼容性。

2. $P_{\text{input}}$ (或 $P_{\text{input},e}$, $P_{\text{input},d}$) 是直接训练的参数，不通过 MLP 生成。

3. MLP 的输出 $[P_K; P_V]$ (或 $[P_{K,e}; P_{V,e}]$, $[P_{K,d}; P_{V,d}]$) 的维度是 $\mathbb{R}^{m \times 2d}$，然后被分割成两个 $\mathbb{R}^{m \times d}$ 的矩阵用于 K 和 V。

4. 在 Encoder-Decoder 模型中，编码器和解码器可以有不同的前缀和 MLP 参数。

5. 这些操作在每一层都会重复应用，但通常使用相同的 MLP 权重。

6. $P_{\text{low}}$ (或 $P_{\text{low},e}$, $P_{\text{low},d}$) 是低维可训练参数，用于生成 K 和 V 的前缀。


### 2. Prompt Tuning

**基本原理：** Prompt Tuning 直接学习连续的软提示，无需 MLP 降维。

| 步骤 | 描述 | 数学表达式 |
|------|------|------------|
| 数学表示 | 软提示 | $P = [p_1, p_2, ..., p_m] \in \mathbb{R}^{m \times d}$ |
| 输入处理 | 拼接操作 | $X' = [P; X] = [p_1, ..., p_m, x_1, ..., x_n]$ |
| 训练过程 | 前向传播 | 将 $X'$ 输入模型 |
| | 反向传播 | 仅计算 $\frac{\partial L}{\partial P}$ |
| | 参数更新 | 只更新 $P$ |

### 3. P-Tuning

**基本原理：** P-Tuning 使用 LSTM 和 MLP 动态生成提示。

| 步骤 | 描述 | 数学表达式 |
|------|------|------------|
| 数学表示 | 可学习嵌入 | $E = [e_1, e_2, ..., e_m] \in \mathbb{R}^{m \times d}$ |
| | LSTM | $LSTM: \mathbb{R}^{m \times d} \rightarrow \mathbb{R}^{m \times h}$ |
| | MLP | $MLP: \mathbb{R}^{h} \rightarrow \mathbb{R}^{d}$ |
| 提示生成过程 | LSTM 处理 | $H = LSTM(E)$ |
| | MLP 映射 | $P_i = MLP(H_i)$, $i \in [1, m]$ |
| | 最终提示 | $P = [P_1, P_2, ..., P_m] \in \mathbb{R}^{m \times d}$ |
| 训练过程 | 反向传播 | 计算 $\frac{\partial L}{\partial E}, \frac{\partial L}{\partial \theta_{LSTM}}, \frac{\partial L}{\partial \theta_{MLP}}$ |
| | 参数更新 | 更新 $E$, LSTM 参数, MLP 参数 |

### 比较和总结

| 方法 | 提示生成 | 参数效率 | 复杂度 |
|------|----------|----------|--------|
| Prefix-Tuning | MLP | 高 | 中 |
| Prompt Tuning | 直接优化 | 最高 | 低 |
| P-Tuning | LSTM + MLP | 中 | 高 |

这三种方法都属于 P 系列微调，它们通过不同方式生成和优化输入序列的一部分来适应特定任务，同时保持模型主体不变，从而实现高效的监督微调。

## 2. LoRA (Low-Rank Adaptation)

### 2.1 基本原理

LoRA 是一种参数高效的微调方法，核心思想是使用低秩矩阵来近似权重更新。

### 2.2 数学表示

假设预训练模型的权重矩阵为 $W \in \mathbb{R}^{d \times k}$，LoRA 的更新可表示为：

$W' = W + \alpha BA$

其中：
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ 是一个小的秩
- $\alpha$ 是可调节的缩放参数

### 2.3 参数效率

| 方法 | 参数数量 |
|------|----------|
| 原始权重矩阵 | $dk$ |
| LoRA | $r(d+k)$ |

通常 $r(d+k) \ll dk$，大幅减少了参数量。

### 2.4 训练过程

| 步骤 | 描述 | 数学表达式 |
|------|------|------------|
| 1. 前向传播 | 计算输出 | $y = (W + \alpha BA)x$ |
| 2. 反向传播 | 计算梯度 | $\frac{\partial L}{\partial A} = \alpha \frac{\partial L}{\partial y} x^T B^T$ |
|              |          | $\frac{\partial L}{\partial B} = \alpha \frac{\partial L}{\partial y} x^T A^T$ |
| 3. 参数更新 | 更新 A 和 B | $A \leftarrow A - \eta \frac{\partial L}{\partial A}$ |
|              |            | $B \leftarrow B - \eta \frac{\partial L}{\partial B}$ |

### 2.5 应用范围

LoRA 通常应用于模型的多个层，特别是注意力层的查询和值矩阵。

### 2.6 优势

1. 参数效率：显著减少训练参数数量
2. 灵活性：易于切换不同任务适应
3. 性能：接近全参数微调的效果

### 2.7 数学直觉

LoRA 基于以下假设：
- 大型预训练模型的权重更新通常位于低维子空间
- 低秩矩阵可有效捕获这种低维结构

这种方法在保持模型性能的同时，大大减少了需要训练和存储的参数数量。

## 3. Adapter-Tuning

Adapter-Tuning 可以视为在 Transformer 中添加特殊的 MLP 层，以实现参数高效的微调。

### 3.1 Adapter 结构

| 组件 | 描述 | 数学表达式 |
|------|------|------------|
| 输入 | 原始层输出 | $x \in \mathbb{R}^d$ |
| 下投影 | 降维操作 | $W_{\text{down}} \in \mathbb{R}^{r \times d}$ |
| 激活函数 | 非线性变换 | $\text{ReLU}$ |
| 上投影 | 升维操作 | $W_{\text{up}} \in \mathbb{R}^{d \times r}$ |
| 残差连接 | 添加原始输入 | $+x$ |
| 输出 | Adapter 输出 | $\text{Adapter}(x) = x + W_{\text{up}}(\text{ReLU}(W_{\text{down}}x))$ |

注：$d$ 为模型隐藏状态维度，$r$ 为 Adapter 瓶颈维度（通常 $r \ll d$）。

### 3.2 Encoder 和 Decoder 加入 Adapter 后的结构

| Encoder 层                   | Decoder 层                   |
|------------------------------|------------------------------|
| 输入                         | 输入                         |
| 自注意力                     | 自注意力 (带掩码)            |
| 残差连接 + 层归一化          | 残差连接 + 层归一化          |
| Adapter 1                    | Adapter 1                    |
| -                            | 交叉注意力                   |
| -                            | 残差连接 + 层归一化          |
| -                            | Adapter 2                    |
| 前馈网络                     | 前馈网络                     |
| 残差连接 + 层归一化          | 残差连接 + 层归一化          |
| Adapter 2                    | Adapter 3                    |
| 输出                         | 输出                         |

注意事项：
1. Encoder 中有 2 个 Adapter，Decoder 中有 3 个 Adapter。
2. Decoder 比 Encoder 多了交叉注意力层及其对应的 Adapter。
3. 每个 Adapter 的结构相同，但参数不同。
4. 原始 Transformer 参数在微调时保持冻结，只更新 Adapter 参数。



### 3.3 参数比较

| 项目 | Transformer | Adapter |
|------|-------------|---------|
| 参数量 | $O(d^2)$ | $O(2dr)$ |
| 训练状态 | 冻结 | 可训练 |
| 任务特异性 | 通用 | 任务特定 |

### 3.4 训练和推理

| 阶段 | 操作 |
|------|------|
| 训练 | 1. 冻结预训练 Transformer 参数<br>2. 初始化 Adapter 参数<br>3. 仅更新 Adapter 参数 |
| 推理 | 1. 选择任务特定 Adapter<br>2. 在 Transformer 计算流程中插入 Adapter |


## 4. BitFit (Bias-terms Fine-tuning)

BitFit 是一种参数高效的微调方法，专注于只调整预训练语言模型中的偏置参数。这种方法旨在通过最小化可训练参数的数量来实现快速和高效的模型适应。

### 4.1 核心思想

- 仅微调模型中现有的偏置参数
- 保持所有权重矩阵固定
- 极大地减少可训练参数数量（通常<1%的总参数）

### 4.2 BitFit 在不同模型结构中的应用

| 组件 | 偏置参数 | 维度 | 适用模型 |
|------|----------|------|----------|
| 自注意力输出 | $b_O$ | $\mathbb{R}^d$ | 所有 |
| 层归一化 | $\beta$ | $\mathbb{R}^d$ | 所有 |
| 前馈网络第一层 | $b_1$ | $\mathbb{R}^{4d}$ | 所有 |
| 前馈网络第二层 | $b_2$ | $\mathbb{R}^d$ | 所有 |
| 交叉注意力输出 | $b_{O_{\text{cross}}}$ | $\mathbb{R}^d$ | 仅 Decoder |
| 分类头 | $b_{\text{cls}}$ | $\mathbb{R}^{|\text{vocab}|}$ | 仅 BERT 类 |


### 4.3 BitFit 的优势

1. **极高的参数效率**：只调整少量偏置参数，大幅减少可训练参数数量
2. **实现简单**：易于在现有模型上实施，无需复杂的架构修改
3. **训练速度快**：由于可训练参数少，训练过程更快
4. **存储效率高**：每个任务只需保存少量参数，便于部署多个任务特定模型

### 4.4 BitFit 的局限性

1. **表达能力有限**：仅调整偏置可能不足以适应复杂任务
2. **性能上限**：在某些任务上可能无法达到全参数微调的性能水平
3. **任务适应性**：不是所有任务都适合仅通过调整偏置来完成
4. **大规模实用性受限**：
   - 对于需要深度语义理解的复杂任务效果可能不佳
   - 在大规模语言模型（如 GPT-3）上的效果尚未得到充分验证
   - 可能不适用于需要模型学习新知识或大幅改变其行为的场景


## 5. Diff-Tuning 在 Transformer 中的应用

| 组件 | 子组件 | 参数 | Diff-Tuning 应用 |
|------|--------|------|-------------------|
| 词嵌入层 | 词嵌入 | $W_e \in \mathbb{R}^{\text{vocab} \times d}$ | $W_e + \Delta W_e$ |
| | 位置编码 | $PE \in \mathbb{R}^{\text{max\_len} \times d}$ | 通常固定，不参与微调 |
| 自注意力层 | 查询/键/值 | $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ | $W_Q + \Delta W_Q$ |
| | | | $W_K + \Delta W_K$ |
| | | | $W_V + \Delta W_V$ |
| | 输出投影 | $W_O \in \mathbb{R}^{d \times d}$ | $W_O + \Delta W_O$ |
| | | $b_O \in \mathbb{R}^d$ | $b_O + \Delta b_O$ |
| 前馈网络层 | 第一层 | $W_1 \in \mathbb{R}^{d \times 4d}$ | $W_1 + \Delta W_1$ |
| | | $b_1 \in \mathbb{R}^{4d}$ | $b_1 + \Delta b_1$ |
| | 第二层 | $W_2 \in \mathbb{R}^{4d \times d}$ | $W_2 + \Delta W_2$ |
| | | $b_2 \in \mathbb{R}^d$ | $b_2 + \Delta b_2$ |
| 层归一化 | 缩放和偏移 | $\gamma \in \mathbb{R}^d$ | $\gamma + \Delta \gamma$ |
| | | $\beta \in \mathbb{R}^d$ | $\beta + \Delta \beta$ |

注意事项：
1. $d$ 是模型的隐藏状态维度
2. 所有的 $\Delta$ 参数都受到约束：$\|\Delta\| \leq \epsilon$，其中 $\epsilon$ 是一个小常数
3. 优化目标：$\min_{\Delta} \mathcal{L}(\theta_{\text{pre-trained}} + \Delta) + \lambda \|\Delta\|$
4. $\lambda$ 是正则化系数，控制参数变化的幅度


[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)
