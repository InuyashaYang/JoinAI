

[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

# 位置编码

## 1. 位置编码的要求

位置编码在序列模型中扮演着至关重要的角色。一个理想的位置编码应满足以下要求：

| 性质 | 描述 | 意义 |
|------|------|------|
| 唯一性 | 每个位置输出一个唯一的编码 | 使模型能够区分和定位序列中的每个元素 |
| 外推性 | 具备良好的外推能力，可处理未见过的位置 | 允许模型处理变长序列和超出训练范围的位置 |
| 相对位置表示 | 两位置之间的差异性只和其相对位置k有关 | 使模型能学习位置无关的模式和关系 |
| 距离相关性 | 相对位置k差越大，两位置之关联越弱 | 反映了语言的局部性原理，近距离词通常关系更密切 |

### 1.1 位置向量之间的运算

我们定义 $f(i, j)$ 为位置 $i$ 和位置 $j$ 之间的注意力交互函数：

$f(i, j) = ((x_i + p_i)W_q)^T ((x_j + p_j)W_k)$

其中：
- $x_i$, $x_j$ 是词向量
- $p_i$, $p_j$ 是位置编码
- $W_q$, $W_k$ 是权重矩阵

展开 $f(i, j)$：

$f(i, j) = (x_i W_q + p_i W_q)^T (x_j W_k + p_j W_k)$

$\quad\;\; = (x_i W_q)^T (x_j W_k) + (x_i W_q)^T (p_j W_k) + (p_i W_q)^T (x_j W_k) + (p_i W_q)^T (p_j W_k)$

$\quad\;\; = x_i^T W_q^T W_k x_j + x_i^T W_q^T W_k p_j + p_i^T W_q^T W_k x_j + p_i^T W_q^T W_k p_j$

#### 解释

完全展开后的 $f(i, j)$ 包含四个项：

| 项 | 数学表达式 | 名称 | 描述 |
|:--:|:----------:|:----:|:-----|
| 1 | $x_i^T W_q^T W_k x_j$ | 纯词向量交互 | 捕捉位置 $i$ 和 $j$ 的词向量之间的语义关系 |
| 2 | $x_i^T W_q^T W_k p_j$ | 词向量-位置编码交互 | 允许模型考虑位置 $i$ 的词与位置 $j$ 的相对位置关系 |
| 3 | $p_i^T W_q^T W_k x_j$ | 位置编码-词向量交互 | 考虑位置 $i$ 相对于位置 $j$ 的词的影响 |
| 4 | $p_i^T W_q^T W_k p_j$ | 纯位置编码交互 | 捕捉两个位置之间的纯粹相对位置关系 |

## 2. 位置编码方法比较

### 2.1 绝对位置编码（Sinusoidal Position Encoding）

#### 2.1.1 定义

| 参数 | 定义 |
|------|------|
| $pos$ | 位置索引 |
| $i$ | 维度索引 |
| $d_{model}$ | 模型维度 |

| 编码类型 | 公式 |
|----------|------|
| 偶数维度 | $PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$ |
| 奇数维度 | $PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$ |

#### 2.1.2 相对位置表示

对于每对 $(2i, 2i+1)$ 维度：

$\begin{bmatrix} 
PE_{(pos+k, 2i)} \\
PE_{(pos+k, 2i+1)}
\end{bmatrix} = 
\begin{bmatrix} 
\cos(k\omega_i) & -\sin(k\omega_i) \\
\sin(k\omega_i) & \cos(k\omega_i)
\end{bmatrix}
\begin{bmatrix} 
PE_{(pos, 2i)} \\
PE_{(pos, 2i+1)}
\end{bmatrix}$

其中 $\omega_i = 1/10000^{2i/d_{model}}$

### 2.2 RoPE (Rotary Position Embedding)

#### 2.2.1 定义

| 参数 | 定义 |
|------|------|
| $m$ | 位置索引 |
| $i$ | 维度索引 |
| $d$ | 模型维度 |
| $\theta_i$ | $10000^{-2i/d}$ |

| 向量 | 定义 |
|------|------|
| $\mathbf{q}_m$ | $[q_0, q_1, ..., q_{d-1}]$ |
| $\mathbf{k}_m$ | $[k_0, k_1, ..., k_{d-1}]$ |

RoPE 操作：

$R_{\Theta,m}(\mathbf{q}_m)_i = [q_i \cos(m\theta_i) - q_{i+1} \sin(m\theta_i), q_i \sin(m\theta_i) + q_{i+1} \cos(m\theta_i)]$

#### 2.2.2 相对位置表示

$\langle R_{\Theta,m}(\mathbf{q}), R_{\Theta,n}(\mathbf{k}) \rangle = f(\mathbf{q}, \mathbf{k}, n-m)$


### 2.3 ALiBi (Attention with Linear Biases)

ALiBi 通过在注意力分数中添加线性偏置来编码位置信息，实际上替换了传统注意力计算中的位置编码相关项。

#### 2.3.1 定义

| 参数 | 定义 |
|------|------|
| $q$ | 查询向量 |
| $k$ | 键向量 |
| $i, j$ | 查询和键的位置索引 |
| $m$ | 预定义斜率 |

#### 2.3.2 注意力计算

ALiBi 修改了传统的注意力计算方式。对比传统方法和 ALiBi：

| 方法 | 注意力分数计算 |
|------|----------------|
| 传统方法 | $a(q, k) = (x_i W_q)^T (x_j W_k) + (x_i W_q)^T (p_j W_k) + (p_i W_q)^T (x_j W_k) + (p_i W_q)^T (p_j W_k)$ |
| ALiBi | $ a(q, k) = q^T k - m ×\text{abs}(i-j) $ |

其中，ALiBi 的 $q^T k$ 对应传统方法中的 $x_i^T W_q^T W_k x_j$（纯词向量交互），而 $- m|i-j|$ 项替代了传统方法中的其他三项（词向量-位置编码交互、位置编码-词向量交互和纯位置编码交互）。

#### 2.3.3 关键特性

1. **简化计算**：ALiBi 通过一个简单的线性项替代了复杂的位置编码交互。
2. **直接编码相对位置**：$|i-j|$ 项直接表示了查询和键之间的相对距离。
3. **消除显式位置编码**：不再需要为输入序列添加单独的位置编码。

这种方法有效地将位置信息整合到注意力机制中，同时简化了计算过程。通过替换传统注意力计算中的位置编码相关项，ALiBi 提供了一种更直接、更高效的方式来处理序列中的位置信息。

### 2.4 比较分析

| 方法 | 优点 | 缺点 |
|------|------|------|
| Sinusoidal | 可以处理任意长度序列；具有良好的数学性质 | 可能难以捕捉非常长的依赖关系 |
| RoPE | 自然地表示相对位置；计算效率高 | 实现相对复杂 |
| ALiBi | 简单有效；易于实现和理解 | 可能不适合某些需要复杂位置关系的任务 |

## 3. 位置编码方法在大型语言模型中的应用



| 模型 | 位置编码方法 | 备注 |
|------|--------------|------|
| GPT-2 | 学习的绝对位置嵌入 | 每个位置都有一个可学习的嵌入向量 |
| GPT-3 | 学习的绝对位置嵌入 | 与 GPT-2 类似，但扩展到更长的序列 |
| BERT | 学习的绝对位置嵌入 | 类似于 GPT-2，但用于双向注意力 |
| T5 | 相对位置编码 | 使用相对位置偏置而不是绝对位置 |
| XLNet | 相对位置编码 | 使用双向相对位置编码 |
| ALBERT | 学习的绝对位置嵌入 | 与 BERT 类似 |
| RoBERTa | 学习的绝对位置嵌入 | 与 BERT 类似，但移除了位置嵌入的可学习性 |
| DeBERTa | 相对位置编码 | 使用解耦的注意力机制 |
| PaLM | RoPE (Rotary Position Embedding) | 使用旋转位置编码 |
| LLaMA | RoPE (Rotary Position Embedding) | 使用旋转位置编码 |
| BLOOM | ALiBi (Attention with Linear Biases) | 使用线性偏置注意力 |
| ERNIE 3.0 | 相对位置编码 | 使用相对位置表示 |
| Falcon | ALiBi (Attention with Linear Biases) | 使用线性偏置注意力 |
| Vicuna | RoPE (Rotary Position Embedding) | 基于 LLaMA，使用旋转位置编码 |
| Alpaca | RoPE (Rotary Position Embedding) | 基于 LLaMA，使用旋转位置编码 |
| BART | 学习的绝对位置嵌入 | 类似于 BERT，用于序列到序列任务 |



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
