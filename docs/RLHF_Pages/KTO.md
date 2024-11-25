
# KTO（Kahneman-Tversky Optimization）：效用的引入与应用

## 前言

**KTO（Kahneman-Tversky Optimization）** 是在 DPO 基础上进一步优化，通过引入效用函数，更加贴近人类决策中的效用感知和偏好。本文将重点讲解 KTO 中效用是如何产生和应用的。

## 1. KTO 中的效用概念

### 1.1 什么是效用？

**效用（Utility）** 是用来衡量某个输出结果对用户的实际价值或满足感。在人类决策过程中，不同的结果会带来不同的效用，这种效用不仅取决于结果本身，还取决于相对于某个参考点的变化。

### 1.2 为什么引入效用？

DPO 主要通过偏好的对数概率比来优化模型，但它没有直接考虑输出结果对用户的实际效用。KTO 通过引入效用函数，使模型优化过程更加符合人类的决策心理，提高生成内容的质量和用户满意度。

## 2. 效用在 KTO 中的来源

### 2.1 前景理论基础

KTO 基于 Kahneman 和 Tversky 的**前景理论**，该理论描述了人类在面对不确定性时的决策行为，强调了**损失厌恶**和**参考点**的重要性。前景理论指出，人们对收益和损失的感知是相对于一个参考点的，而不是绝对值。

### 2.2 定义效用函数

在 KTO 中，效用函数 $v(x, y)$ 被定义为：

$$
v(x, y) =
\begin{cases}
\lambda_D \cdot \sigma\left(\beta \cdot (r_\theta(x, y) - z_0)\right) & \text{如果 } y \text{ 是可取的输出} \\
\lambda_U \cdot \sigma\left(\beta \cdot (z_0 - r_\theta(x, y))\right) & \text{如果 } y \text{ 是不可取的输出}
\end{cases}
$$

其中：
- $\sigma$ 是 sigmoid 函数，将效用值限制在 0 到 1 之间。
- $\beta$ 控制效用函数的敏感度。
- $\lambda_D$ 和 $\lambda_U$ 分别是对可取和不可取输出的权重，用于处理数据不平衡。
- $r_\theta(x, y) = \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是策略模型相对于参考模型的对数概率比。
- $z_0$ 是参考点，通常设定为参考模型的期望奖励。

## 3. 效用的计算与应用

### 3.1 计算参考点 $z_0$

参考点 $z_0$ 代表了参考模型的期望表现，在 KTO 中通过以下方法估计：

$$
z_0 = \mathbb{E}_{y' \sim \pi_{\text{ref}}(Y|x)}[r_\theta(x, y')]
$$

由于直接计算期望值较为复杂，KTO 采用批次内不匹配对的方法进行近似估计：

$$
\hat{z}_0 = \max\left(0, \frac{1}{m} \sum_{i=1}^m \log\frac{\pi_\theta(y_{j}|x_i)}{\pi_{\text{ref}}(y_{j}|x_i)}\right)
$$

其中，$m$ 是批次大小，$j = (i + 1) \mod m$ 表示在同一批次内循环配对不同的输入和输出。

### 3.2 计算效用值

根据输出 $y$ 的类型（可取或不可取），计算对应的效用值：

- **可取输出 $y_w$**：
  
  $$
  v(x, y_w) = \lambda_D \cdot \sigma\left(\beta \cdot (r_\theta(x, y_w) - z_0)\right)
  $$

- **不可取输出 $y_l$**：
  
  $$
  v(x, y_l) = \lambda_U \cdot \sigma\left(\beta \cdot (z_0 - r_\theta(x, y_l))\right)
  $$

### 3.3 总效用计算

结合可取和不可取输出的效用，定义总效用 $U(x, y_w, y_l)$：

$$
U(x, y_w, y_l) = v(x, y_w) - v(x, y_l)
$$

## 4. 损失函数设计

KTO 的损失函数旨在最大化生成内容的总效用，具体定义为：

$$
\mathcal{L}_{\text{KTO}} = -\log(\sigma(U(x, y_w, y_l)))
$$

替换总效用的表达式，可以进一步展开为：

$$
\mathcal{L}_{\text{KTO}} = -\log\left(\sigma\left(\lambda_D \cdot \sigma\left(\beta \cdot (r_\theta(x, y_w) - z_0)\right) - \lambda_U \cdot \sigma\left(\beta \cdot (z_0 - r_\theta(x, y_l))\right)\right)\right)
$$

## 5. 训练步骤总结

### 5.1 初始化

1. **预训练模型**：使用大规模语料库预训练语言模型，最大化下一个 token 的对数似然。
2. **监督微调（SFT）**：在特定任务和指令数据上进行微调，提升模型的指令响应能力。
3. **设置参考模型**：通常选择经过 SFT 的模型作为参考模型 $\pi_{\text{ref}}$。

### 5.2 数据准备

1. **构建偏好对**：将用户偏好转换为二元信号 $(x, y_w, y_l)$，其中 $y_w$ 为可取输出，$y_l$ 为不可取输出。
2. **调整权重**：根据可取和不可取输出的比例，设置 $\lambda_D$ 和 $\lambda_U$ 以处理数据不平衡。

### 5.3 前向传播与效用计算

1. **计算对数概率比**：

   $$
   r_\theta(x, y) = \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
   $$

2. **估计参考点 $z_0$**：通过批次内不匹配对方法计算 $\hat{z}_0$。
3. **计算效用值**：根据输出类型计算 $v(x, y_w)$ 和 $v(x, y_l)$。
4. **计算总效用 $U(x, y_w, y_l)$**。

### 5.4 损失计算与优化

1. **计算损失**：

   $$
   \mathcal{L}_{\text{KTO}} = -\log(\sigma(U(x, y_w, y_l)))
   $$

2. **反向传播与参数更新**：使用优化器（如 AdamW）最小化损失，更新模型参数 $\theta$。

### 5.5 迭代训练

重复上述步骤，持续优化模型，使其生成内容的效用最大化，逐步提升模型的对齐效果。

## 6. 超参数设置

- **学习率 $\beta$**：控制效用函数的敏感度，推荐从 $5 \times 10^{-6}$ 开始，根据效果调整。
- **批量大小**：至少为 2，用于参考点的估计。推荐范围为 8 到 128。
- **权重参数 $\lambda_D$ 和 $\lambda_U$**：根据可取和不可取输出的比例设置，建议 $\lambda_D \cdot n_D$ 和 $\lambda_U \cdot n_U$ 保持在 [1, 1.5] 之间。

## 7. KTO 与 DPO 的关键区别

| **特性**            | **DPO**                                | **KTO**                                        |
|---------------------|----------------------------------------|------------------------------------------------|
| **理论基础**        | 最大化偏好的对数概率比                | 基于前景理论，最大化效用                      |
| **损失函数设计**    | 基于概率比率                           | 引入效用函数和参考点，结合权重参数            |
| **优化目标**        | 提升偏好输出的概率                    | 直接优化生成内容的效用，更符合人类决策心理      |
| **参考点处理**      | 通过对数概率比定义                     | 显式引入效用参考点 $z_0$，基于批次内不匹配对 |



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
