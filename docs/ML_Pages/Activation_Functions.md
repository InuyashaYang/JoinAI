
## 1. Sigmoid 函数

Sigmoid 函数是最早被广泛使用的激活函数之一，其数学表达式为：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

### 特点

- **输出范围**：$(0, 1)$，非常适合用于二分类问题的输出层。
- **平滑性**：Sigmoid 函数是一个平滑且连续的函数，具有良好的导数性质。
- **缺点**：
  - **梯度消失**：当输入值绝对值较大时，梯度趋近于零，导致训练缓慢。
  - **输出不以零为中心**：这可能会影响梯度下降的效率。

## 2. 双曲正切（Tanh）函数

Tanh 函数是 Sigmoid 函数的一个变种，其输出范围为 $(-1, 1)$。数学表达式为：

$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

### 特点

- **输出范围**：$(-1, 1)$，相比 Sigmoid 更加以零为中心，有助于加快收敛速度。
- **平滑性**：同样是平滑且连续的函数。
- **缺点**：
  - **梯度消失**：与 Sigmoid 类似，在输入值较大或较小时，梯度会趋近于零。

## 3. 修正线性单元（ReLU）

ReLU（Rectified Linear Unit）是当前最为流行的激活函数之一，定义为：

$$
\text{ReLU}(x) = \max(0, x)
$$

### 特点

- **输出范围**：$[0, +\infty)$。
- **计算简单**：计算速度快，适合大规模神经网络。
- **缓解梯度消失**：在正区间，梯度恒为 $1$，有效缓解了梯度消失问题。
- **缺点**：
  - **"死亡 ReLU" 问题**：在负区间，梯度为 $0$，可能导致部分神经元在训练中“死亡”，长期不更新。

## 4. 带泄漏的 ReLU（Leaky ReLU）

为了解决 ReLU 的“死亡”问题，引入了带泄漏的 ReLU，其定义为：

$$
\text{Leaky ReLU}(x) = \begin{cases}
x & \text{if } x \geq 0 \\
\alpha x & \text{if } x < 0
\end{cases}
$$

其中，$\alpha$ 是一个小的常数（如 $0.01$）。

### 特点

- **输出范围**：$(-\infty, +\infty)$。
- **避免“死亡”**：即使在负区间，仍有一部分梯度传递，减少神经元“死亡”的可能性。
- **缺点**：
  - **选择参数**：需要为 $\alpha$ 选择一个合适的值，通常需要通过实验确定。

## 5. 参数化 ReLU（PReLU）

PReLU（Parametric ReLU）是对 Leaky ReLU 的扩展，其泄漏系数 $\alpha$ 是可学习的参数：

$$
\text{PReLU}(x) = \begin{cases}
x & \text{if } x \geq 0 \\
\alpha x & \text{if } x < 0
\end{cases}
$$

### 特点

- **自适应**：$\alpha$ 可以在训练过程中自动调整，增强了模型的表达能力。
- **缺点**：
  - **增加参数**：引入了额外的参数，可能增加模型的复杂性。

## 6. 指数线性单元（ELU）

ELU（Exponential Linear Unit）在负区间引入了指数形式，使得函数具有更好的性能。定义为：

$$
\text{ELU}(x) = \begin{cases}
x & \text{if } x \geq 0 \\
\alpha (e^{x} - 1) & \text{if } x < 0
\end{cases}
$$

其中，$\alpha$ 通常为 $1$。

### 特点

- **输出范围**：$(-\alpha, +\infty)$。
- **缓解梯度问题**：在负区间，梯度不为零，帮助模型学习。
- **平滑性**：相比 ReLU 和 Leaky ReLU 更加平滑，有助于提升模型性能。
- **缺点**：
  - **计算复杂度**：相比 ReLU，计算稍显复杂。

## 7. 滑动空间单元（Swish）

Swish 函数由 Google 提出，是一种自门控的激活函数，定义为：

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

### 特点

- **输出范围**：$(-\infty, +\infty)$。
- **平滑且非饱和**：有助于信息的流动和梯度的传递。
- **性能优越**：在某些任务上，Swish 超越了 ReLU 和其他激活函数的表现。
- **缺点**：
  - **计算复杂度**：需要额外的计算 Sigmoid 函数，增加了计算成本。

## 8. Softmax 函数

Softmax 函数通常用于多分类问题的输出层，其定义为：

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

### 特点

- **概率分布**：将输出转化为概率分布，所有输出值和为 $1$。
- **适用场景**：多分类任务，如图像分类、文本分类等。
- **缺点**：
  - **数值稳定性**：在计算时可能出现数值溢出，需要进行数值稳定性处理（如减去最大值）。

## 总结

激活函数在神经网络中起到了至关重要的作用，通过引入非线性，使得网络能够逼近复杂的函数。选择合适的激活函数取决于具体的任务和网络结构。以下是常见激活函数的对比：

| 激活函数 | 公式 | 输出范围 | 优点 | 缺点 |
|----------|------|----------|------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ | 平滑、连续 | 梯度消失、输出不以零为中心 |
| Tanh | $\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$ | $(-1, 1)$ | 以零为中心 | 梯度消失 |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | $[0, +\infty)$ | 计算简单、缓解梯度消失 | 死亡 ReLU 问题 |
| Leaky ReLU | $\text{Leaky ReLU}(x) = \max(0, x) + \alpha \min(0, x)$ | $(-\infty, +\infty)$ | 减少死亡问题 | 需要选择参数 $\alpha$ |
| PReLU | 同 Leaky ReLU，但 $\alpha$ 可学习 | $(-\infty, +\infty)$ | 自适应调整 $\alpha$ | 增加模型复杂性 |
| ELU | $\text{ELU}(x) = \begin{cases} x & x \geq 0 \\ \alpha (e^{x} - 1) & x < 0 \end{cases}$ | $(-\alpha, +\infty)$ | 缓解梯度问题、平滑 | 计算复杂度 |
| Swish | $\text{Swish}(x) = x \cdot \sigma(x)$ | $(-\infty, +\infty)$ | 性能优越、平滑 | 计算复杂度 |
| Softmax | $\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$ | $(0, 1)$，和为 $1$ | 转化为概率分布 | 数值稳定性 |



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
