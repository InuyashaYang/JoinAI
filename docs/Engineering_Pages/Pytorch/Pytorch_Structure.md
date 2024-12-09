[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/AIDIY?style=social)](https://github.com/InuyashaYang/AIDIY)

[Pytorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)

## 主要类列表

1. **`Tensor` 类**
   - **职责**：
     - 存储数据（使用 NumPy 的 `ndarray`）和相应的梯度。
     - 记录创建该张量的操作，便于构建计算图。
   - **主要属性**：
     - `data`: 存储实际的数值数据。
     - `grad`: 存储相对于损失函数的梯度。
     - `creator`: 记录执行当前张量计算的操作节点。

2. **`Operation` 类（基类）**
   - **职责**：
     - 定义所有操作节点的基础接口，包括前向计算和反向传播的方法。
   - **主要方法**：
     - `forward(inputs)`: 执行前向计算，生成输出张量。
     - `backward(gradient)`: 执行反向传播，计算输入张量的梯度。

3. **具体操作类（继承自 `Operation`）**
   - **职责**：
     - 实现具体的数学操作及其对应的梯度计算。
   - **主要子类**：
     - **`Add`**：
       - **功能**：实现加法运算。
       - **前向**：`output = input_a + input_b`
       - **反向**：`grad_a = grad_output * 1`, `grad_b = grad_output * 1`
     - **`Multiply`**：
       - **功能**：实现逐元素乘法运算。
       - **前向**：`output = input_a * input_b`
       - **反向**：`grad_a = grad_output * input_b`, `grad_b = grad_output * input_a`
     - **`MatMul`**：
       - **功能**：实现矩阵乘法运算。
       - **前向**：`output = input_a @ input_b`
       - **反向**：`grad_a = grad_output @ input_b.T`, `grad_b = input_a.T @ grad_output`
     - **`ReLU`**：
       - **功能**：实现 ReLU 激活函数。
       - **前向**：`output = max(0, input)`
       - **反向**：`grad_input = grad_output * (input > 0)`
     - **其他操作**：
       - 根据需要，可以实现更多操作，如 `Sigmoid`、`Softmax`、`Transpose` 等。

4. **`Graph` 类**
   - **职责**：
     - 管理和维护整个计算图的结构。
     - 组织节点和边，确保前向和反向传播的顺序。
   - **主要方法**：
     - `add_operation(operation)`: 添加操作节点到计算图中。
     - `forward(inputs)`: 执行前向传播，计算输出。
     - `backward(loss_grad)`: 执行反向传播，计算梯度。

5. **`Optimizer` 类（基类）**
   - **职责**：
     - 定义优化算法的基础接口，用于更新模型参数。
   - **主要方法**：
     - `step()`: 更新参数。
     - `zero_grad()`: 清零所有参数的梯度。

6. **具体优化器类（继承自 `Optimizer`）**
   - **职责**：
     - 实现具体的优化算法。
   - **主要子类**：
     - **`SGD`**（随机梯度下降）：
       - **功能**：按学习率更新参数。
       - **实现**：`param -= learning_rate * param.grad`
     - **`Adam`**（可选）：
       - **功能**：实现 Adam 优化算法。
       - **实现**：包括动量和自适应学习率的计算。

7. **`Layer` 类**
   - **职责**：
     - 抽象神经网络中的层，管理参数和前向计算。
   - **主要方法**：
     - `forward(inputs)`: 定义层的前向传播。
     - `parameters()`: 返回层的可训练参数。
   - **主要子类**：
     - **`Linear`**：
       - **功能**：实现全连接层。
       - **参数**：权重矩阵和偏置向量。
       - **前向**：`output = input @ weights + bias`
     - **其他层**：
       - 根据需要，可以实现更多层，如 `Conv2D`、`Dropout` 等。

8. **`Model` 类**
   - **职责**：
     - 组合多个 `Layer`，构建整个神经网络模型。
     - 管理整体的前向和反向传播过程。
   - **主要方法**：
     - `forward(inputs)`: 顺序调用各层的前向传播。
     - `parameters()`: 汇总所有层的可训练参数。

9. **`Loss` 类**
   - **职责**：
     - 定义损失函数及其梯度计算。
   - **主要方法**：
     - `forward(predictions, targets)`: 计算损失值。
     - `backward()`: 计算损失相对于预测值的梯度。
   - **主要子类**：
     - **`MeanSquaredError`**：
       - **功能**：实现均方误差损失。
     - **`CrossEntropyLoss`**：
       - **功能**：实现交叉熵损失。

## 类之间的关系图示（简化）

```
Model
├── Layer
│   ├── Linear
│   └── ...
├── Loss
│   ├── MeanSquaredError
│   └── CrossEntropyLoss
└── Optimizer
    ├── SGD
    └── Adam
```

## 进一步的实现步骤

1. **定义 `Tensor` 类**：
   - 实现数据存储、梯度存储以及与操作的连接。
   
2. **实现 `Operation` 基类及其子类**：
   - 每个子类需要实现具体的前向和反向计算逻辑。
   
3. **构建 `Graph` 类**：
   - 管理操作的执行顺序，支持前向和反向传播。
   
4. **设计 `Layer` 和 `Model` 类**：
   - 方便模块化地构建复杂的神经网络结构。
   
5. **实现优化器**：
   - 选择合适的优化算法，确保模型参数能够有效更新。
   
6. **定义损失函数**：
   - 选择合适的损失函数，根据具体任务（如回归或分类）进行选择。

通过以上类的设计与实现，我们将能够搭建一个基本但功能完整的深度学习框架，支持构建计算图、执行前向和反向传播，以及进行参数优化。这将为深入理解深度学习框架的底层原理打下坚实的基础。

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
