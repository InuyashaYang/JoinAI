[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/JoinAI?style=social)](https://github.com/InuyashaYang/JoinAI)

## CNN

### CNN 核心数学概念

| 概念 | 数学表达式 | 说明 |
|------|------------|------|
| 2D 卷积 | $$(I * K)(i,j) = \sum_{m}\sum_{n} I(m,n)K(i-m,j-n)$$ | $I$: 输入图像<br>$K$: 卷积核 |
| ReLU 激活函数 | $$f(x) = \max(0,x)$$ | 非线性激活 |
| ReLU 导数 | $$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$ | 用于反向传播 |
| 最大池化 | $$y_{ij} = \max_{(a,b)\in R_{ij}} x_{ab}$$ | $R_{ij}$: 池化区域 |
| 全连接层 | $$y = Wx + b$$ | $W$: 权重矩阵<br>$b$: 偏置向量 |
| 交叉熵损失 | $$L = -\sum_{i} y_i \log(\hat{y}_i)$$ | $y_i$: 真实标签<br>$\hat{y}_i$: 预测概率 |
| 梯度计算 | $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$$ | 链式法则 |
| 参数更新 | $$w_{t+1} = w_t - \eta \frac{\partial L}{\partial w_t}$$ | $\eta$: 学习率 |
| CNN 层表示 | $$h^{l+1} = f(W^l * h^l + b^l)$$ | $h^l$: 第 $l$ 层特征图<br>$W^l$: 卷积核<br>$b^l$: 偏置 |

### CNN 结构相关计算

| 计算 | 公式 | 参数说明 |
|------|------|----------|
| 感受野大小 | $$r_l = r_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i$$ | $r_l$: 第 $l$ 层感受野大小<br>$k_l$: 第 $l$ 层卷积核大小<br>$s_i$: 第 $i$ 层步长 |
| 输出特征图大小 | $$O = \frac{W - K + 2P}{S} + 1$$ | $O$: 输出大小<br>$W$: 输入大小<br>$K$: 卷积核大小<br>$P$: 填充大小<br>$S$: 步长 |

### CNN 层参数数量

| 层类型 | 参数数量 | 说明 |
|--------|----------|------|
| 卷积层 | $C_{in} \times K_h \times K_w \times C_{out} + C_{out}$ | $C_{in}$: 输入通道数<br>$K_h, K_w$: 卷积核高度和宽度<br>$C_{out}$: 输出通道数 |
| 全连接层 | $N_{in} \times N_{out} + N_{out}$ | $N_{in}$: 输入神经元数<br>$N_{out}$: 输出神经元数 |


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
