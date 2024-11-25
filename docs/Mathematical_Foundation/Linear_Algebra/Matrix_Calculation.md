
## 矩阵求导

### 1. 基本定义

| 类型 | 定义 | 维度 |
|------|------|------|
| 标量对矩阵 | $\frac{\partial f}{\partial \mathbf{X}} = \left[\frac{\partial f}{\partial X_{ij}}\right]$ | $f: \mathbb{R}^{m \times n} \to \mathbb{R}$<br>$\frac{\partial f}{\partial \mathbf{X}} \in \mathbb{R}^{m \times n}$ |
| 矩阵对矩阵 | $\frac{\partial \mathbf{F}}{\partial \mathbf{X}} = \left[\frac{\partial F_{ij}}{\partial X_{kl}}\right]$ | $\mathbf{F}: \mathbb{R}^{m \times n} \to \mathbb{R}^{p \times q}$<br>$\frac{\partial \mathbf{F}}{\partial \mathbf{X}} \in \mathbb{R}^{pq \times mn}$ | 取决于布局约定 |

索引排列说明：


1. 分子布局 (Numerator layout):
   - 排列顺序：$(i,j)$ 先变化（行），$(k,l)$ 后变化（列）
   - 矩阵形式：
     $$\begin{bmatrix}
     \frac{\partial F_{11}}{\partial X_{11}} & \frac{\partial F_{11}}{\partial X_{12}} & \cdots & \frac{\partial F_{11}}{\partial X_{mn}} \\
     \frac{\partial F_{12}}{\partial X_{11}} & \frac{\partial F_{12}}{\partial X_{12}} & \cdots & \frac{\partial F_{12}}{\partial X_{mn}} \\
     \vdots & \vdots & \ddots & \vdots \\
     \frac{\partial F_{pq}}{\partial X_{11}} & \frac{\partial F_{pq}}{\partial X_{12}} & \cdots & \frac{\partial F_{pq}}{\partial X_{mn}}
     \end{bmatrix}$$

2. 分母布局 (Denominator layout):
   - 排列顺序：$(k,l)$ 先变化（行），$(i,j)$ 后变化（列）
   - 矩阵形式：
     $$\begin{bmatrix}
     \frac{\partial F_{11}}{\partial X_{11}} & \frac{\partial F_{12}}{\partial X_{11}} & \cdots & \frac{\partial F_{pq}}{\partial X_{11}} \\
     \frac{\partial F_{11}}{\partial X_{12}} & \frac{\partial F_{12}}{\partial X_{12}} & \cdots & \frac{\partial F_{pq}}{\partial X_{12}} \\
     \vdots & \vdots & \ddots & \vdots \\
     \frac{\partial F_{11}}{\partial X_{mn}} & \frac{\partial F_{12}}{\partial X_{mn}} & \cdots & \frac{\partial F_{pq}}{\partial X_{mn}}
     \end{bmatrix}$$


- 分子布局：每一行对应 $\mathbf{F}$ 的一个元素，每一列对应 $\mathbf{X}$ 的一个元素。
- 分母布局：每一行对应 $\mathbf{X}$ 的一个元素，每一列对应 $\mathbf{F}$ 的一个元素。


3. 混合布局：
   - 有时也使用 $\mathbb{R}^{p \times q \times m \times n}$ 的四维张量表示，避免展平的歧义


### 2. 导数布局

| 布局 | 定义 |
|------|------|
| 分子布局 | $\left[\frac{\partial \mathbf{F}}{\partial \mathbf{X}}\right]_{ij,kl} = \frac{\partial F_{ij}}{\partial X_{kl}}$ |
| 分母布局 | $\left[\frac{\partial \mathbf{F}}{\partial \mathbf{X}}\right]_{kl,ij} = \frac{\partial F_{ij}}{\partial X_{kl}}$ |

### 3. 一般公式

| 函数类型 | 公式 |
|----------|------|
| 线性函数 | $\frac{\partial (\mathbf{AX})}{\partial \mathbf{X}} = \mathbf{A}^T$<br>$\frac{\partial (\mathbf{XA})}{\partial \mathbf{X}} = \mathbf{A}$ |
| 二次型 | $\frac{\partial (\mathbf{X}^T\mathbf{AX})}{\partial \mathbf{X}} = \mathbf{AX} + \mathbf{A}^T\mathbf{X}$ |
| 迹 | $\frac{\partial \text{tr}(\mathbf{AX})}{\partial \mathbf{X}} = \mathbf{A}^T$<br>$\frac{\partial \text{tr}(\mathbf{XAX}^T)}{\partial \mathbf{X}} = \mathbf{X}(\mathbf{A} + \mathbf{A}^T)$ |
| 行列式 | $\frac{\partial \det(\mathbf{X})}{\partial \mathbf{X}} = \det(\mathbf{X})(\mathbf{X}^{-1})^T$ |
| 逆矩阵 | $\frac{\partial \mathbf{X}^{-1}}{\partial \mathbf{X}} = -(\mathbf{X}^{-1})^T \otimes \mathbf{X}^{-1}$ |

### 4. 链式法则

| 情况 | 公式 |
|------|------|
| 标量情况 | $\frac{\partial f}{\partial \mathbf{X}} = \text{tr}\left(\frac{\partial f}{\partial \mathbf{Y}}^T \frac{\partial \mathbf{Y}}{\partial \mathbf{X}}\right)$ |
| 矩阵情况 | $\frac{\partial \mathbf{F}}{\partial \mathbf{X}} = \frac{\partial \mathbf{F}}{\partial \mathbf{Y}} \frac{\partial \mathbf{Y}}{\partial \mathbf{X}}$ |

### 5. 常见神经网络操作的导数

| 操作 | 导数 |
|------|------|
| 线性层<br>$\mathbf{Y} = \mathbf{WX} + \mathbf{b}$ | $\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{X}^T$<br>$\frac{\partial L}{\partial \mathbf{X}} = \mathbf{W}^T \frac{\partial L}{\partial \mathbf{Y}}$<br>$\frac{\partial L}{\partial \mathbf{b}} = \sum_{i} \frac{\partial L}{\partial \mathbf{Y}_i}$ |
| 激活函数<br>$\mathbf{Y} = \sigma(\mathbf{X})$ | $\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \odot \sigma'(\mathbf{X})$ |

注：
- $\otimes$ 表示Kronecker积
- $\odot$ 表示Hadamard（元素wise）乘积
- 在激活函数中，$\sigma'(\mathbf{X})$ 表示激活函数的导数


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
