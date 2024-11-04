# 矩阵的满秩分解详解

### 基本概念

给定矩阵 $A \in \mathbb{R}^{m \times n}$，其秩为 $r$，满秩分解是将 $A$ 表示为：

$A = LR$

其中：
- $L \in \mathbb{R}^{m \times r}$ 为基底矩阵，由 $r$ 个线性无关的列向量组成
- $R \in \mathbb{R}^{r \times n}$ 为系数矩阵，表示原矩阵各列在基底上的线性组合系数

### 方法一：基本步骤法

#### 1. 理论基础

基本步骤法基于以下定理：
- 任意矩阵的列空间可由其线性无关列向量张成
- 矩阵的任意列都可以表示为这些基底向量的线性组合

#### 2. 算法步骤

1) **确定矩阵秩**
   - 通过初等变换将矩阵化为行阶梯形
   - 非零行的数量即为矩阵的秩 $r$

2) **选取基底列向量**
   - 从矩阵 $A$ 中选择 $r$ 个线性无关的列向量
   - 构成基底矩阵 $L \in \mathbb{R}^{m \times r}$

3) **求解系数矩阵**
   - 对每列 $A_j$ 求解方程组：$A_j = Lx$
   - 将解向量 $x$ 作为矩阵 $R$ 的第 $j$ 列

#### 3. 详细示例

考虑矩阵：
$A = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}$

##### 3.1 确定秩
通过行阶梯形变换：
$\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix} \rightarrow 
\begin{pmatrix}
1 & 2 & 3 \\
0 & -3 & -6 \\
0 & 0 & 0
\end{pmatrix}$

得知 $rank(A) = 2$

##### 3.2 选取基底
选择第一、二列作为基底：
$L = \begin{pmatrix}
1 & 2 \\
4 & 5 \\
7 & 8
\end{pmatrix}$

##### 3.3 求解系数
对第三列求解方程组：
$\begin{pmatrix} 3 \\ 6 \\ 9 \end{pmatrix} = x_1\begin{pmatrix} 1 \\ 4 \\ 7 \end{pmatrix} + x_2\begin{pmatrix} 2 \\ 5 \\ 8 \end{pmatrix}$

写成增广矩阵：
$\begin{pmatrix}
1 & 2 & | & 3 \\
4 & 5 & | & 6 \\
7 & 8 & | & 9
\end{pmatrix}$

通过高斯消元解得：$x_1 = -1$, $x_2 = 2$

同理求解其他列，最终得到：
$R = \begin{pmatrix}
1 & 0 & -1 \\
0 & 1 & 2
\end{pmatrix}$

### 方法二：高斯消元法

#### 1. 理论基础

高斯消元法基于以下性质：
- 初等行变换不改变矩阵的列空间
- 行阶梯形中主元列的原始列向量线性无关

#### 2. 算法步骤

1) **矩阵化简**
   - 对矩阵 $A$ 进行行阶梯形变换
   - 记录主元位置的列号 $i_1,\ldots,i_r$

2) **构造基底矩阵**
   - $L = [A_{i_1},\ldots,A_{i_r}]$
   - 其中 $A_{i_k}$ 为原矩阵的第 $i_k$ 列

3) **构造系数矩阵**
   - 利用化简后的矩阵形式
   - 提取线性组合关系构造 $R$

#### 3. 详细示例

考虑同样的矩阵：
$A = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}$

##### 3.1 高斯消元过程

第一步：
$\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix} \rightarrow 
\begin{pmatrix}
1 & 2 & 3 \\
0 & -3 & -6 \\
0 & -6 & -12
\end{pmatrix}$

第二步：
$\begin{pmatrix}
1 & 2 & 3 \\
0 & -3 & -6 \\
0 & -6 & -12
\end{pmatrix} \rightarrow 
\begin{pmatrix}
1 & 2 & 3 \\
0 & -3 & -6 \\
0 & 0 & 0
\end{pmatrix}$

##### 3.2 提取基底
主元在第1、2列，因此：
$L = \begin{pmatrix}
1 & 2 \\
4 & 5 \\
7 & 8
\end{pmatrix}$

##### 3.3 构造系数矩阵
从行阶梯形读取关系：
$R = \begin{pmatrix}
1 & 0 & -1 \\
0 & 1 & 2
\end{pmatrix}$

验证：
$LR = \begin{pmatrix}
1 & 2 \\
4 & 5 \\
7 & 8
\end{pmatrix}
\begin{pmatrix}
1 & 0 & -1 \\
0 & 1 & 2
\end{pmatrix}
= \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}$

### QR分解
```python
import numpy as np

def rank_factorization_qr(A):
    # QR分解
    Q, R = np.linalg.qr(A)
    
    # 确定数值秩
    tol = 1e-10
    r = np.sum(np.abs(np.diag(R)) > tol)
    
    # 构造满秩分解
    L = Q[:, :r]
    R = R[:r, :]
    
    return L, R
```

### SVD分解
```python
import numpy as np

def rank_factorization_svd(A):
    # SVD分解
    U, S, Vh = np.linalg.svd(A)
    
    # 确定数值秩
    tol = 1e-10
    r = np.sum(S > tol)
    
    # 构造满秩分解
    L = U[:, :r] @ np.diag(np.sqrt(S[:r]))
    R = np.diag(np.sqrt(S[:r])) @ Vh[:r, :]
    
    return L, R
```

### 高斯消元分解
```python
import numpy as np

def rank_factorization_gaussian(A):
    # 确保A是浮点数类型
    A = A.astype(np.float64)
    
    m, n = A.shape
    tol = 1e-10
    
    # 转换为行阶梯形
    U = np.copy(A)
    pivots = []
    
    rank = 0
    for j in range(n):
        # 找主元
        max_idx = rank + np.argmax(np.abs(U[rank:, j]))
        if np.abs(U[max_idx, j]) > tol:
            # 交换行
            if max_idx != rank:
                U[rank], U[max_idx] = U[max_idx].copy(), U[rank].copy()
            
            # 消元
            for i in range(rank + 1, m):
                factor = U[i, j] / U[rank, j]
                U[i] -= factor * U[rank]
            
            pivots.append(j)
            rank += 1
    
    # 构造分解
    L = A[:, pivots]
    R = np.linalg.solve(L.T @ L, L.T @ A)
    
    return L, R
```

从计算机的实际开销来看，在处理3x3维矩阵时，这三种方法的执行速度是差不多的