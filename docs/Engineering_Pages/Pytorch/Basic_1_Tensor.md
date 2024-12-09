[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/AIDIY?style=social)](https://github.com/InuyashaYang/AIDIY)

## 一、张量（Tensor）

### 1. 张量的概念

张量（Tensor）是 PyTorch 的核心数据结构，类似于 NumPy 的 `ndarray`，但支持 GPU 加速。张量可以表示标量（0维）、向量（1维）、矩阵（2维）以及更高维度的数据。

### 2. 创建张量

PyTorch 提供了多种方法来创建张量：

#### 2.1 从数据创建张量

使用 `torch.tensor(data)` 从列表、数组等数据创建一个新的张量。

```python
import torch

data = [1, 2, 3, 4, 5]
tensor_data = torch.tensor(data)
print(tensor_data)
# 输出: tensor([1, 2, 3, 4, 5])
```

#### 2.2 创建全零张量

使用 `torch.zeros(size)` 创建一个指定大小的张量，所有元素的值为0。

```python
size = (2, 3)
zeros_tensor = torch.zeros(size)
print(zeros_tensor)
# 输出:
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
```

#### 2.3 创建全一张量

使用 `torch.ones(size)` 创建一个指定大小的张量，所有元素的值为1。

```python
size = (2, 3)
ones_tensor = torch.ones(size)
print(ones_tensor)
# 输出:
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
```

#### 2.4 创建未初始化张量

使用 `torch.empty(size)` 创建一个指定大小的未初始化张量，其值取决于内存的状态。

```python
size = (2, 3)
empty_tensor = torch.empty(size)
print(empty_tensor)
# 输出示例:
# tensor([[1.4013e-45, 0.0000e+00, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00]])
```

#### 2.5 创建服从标准正态分布的张量

使用 `torch.randn(size)` 创建一个指定大小的张量，元素值从标准正态分布中随机抽取。

```python
size = (2, 3)
randn_tensor = torch.randn(size)
print(randn_tensor)
# 输出示例:
# tensor([[ 0.4963, -0.1383,  0.6477],
#         [ 1.4393,  0.3337,  0.4621]])
```

#### 2.6 创建范围内的一维张量

使用 `torch.arange(start, end, step)` 创建一个一维张量，元素值从起始值到结束值，步长为给定的步长。

```python
start = 0
end = 5
step = 1
arange_tensor = torch.arange(start, end, step)
print(arange_tensor)
# 输出: tensor([0, 1, 2, 3, 4])
```

#### 2.7 创建均匀间隔的张量

使用 `torch.linspace(start, end, steps)` 创建一个在指定范围内均匀间隔的一维张量。

```python
start = 0
end = 5
steps = 5
linspace_tensor = torch.linspace(start, end, steps)
print(linspace_tensor)
# 输出: tensor([0.0000, 1.2500, 2.5000, 3.7500, 5.0000])
```

### 3. 张量的属性

#### 3.1 数据类型（dtype）

获取张量中元素的数据类型。

```python
import torch

tensor = torch.tensor([1, 2, 3])
print(tensor.dtype)
# 输出: torch.int64
```

#### 3.2 形状（shape）

获取张量的形状，返回一个 `torch.Size` 对象。

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor.shape)
# 输出: torch.Size([2, 3])
```

#### 3.3 设备（device）

获取张量所在的设备，如 `cpu` 或 `cuda:0`。

```python
tensor = torch.tensor([1, 2, 3])
print(tensor.device)
# 输出: cpu
```

### 4. 索引、切片与拼接

#### 4.1 索引操作

使用索引访问张量中的元素。

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
element = tensor[0, 1]  # 访问第0行第1列的元素
print(element)
# 输出: tensor(2)
```

#### 4.2 切片操作

使用切片获取张量的子张量。

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
sub_tensor = tensor[:, 1:]  # 获取所有行，第1列及之后的所有列
print(sub_tensor)
# 输出:
# tensor([[2, 3],
#         [5, 6]])
```

#### 4.3 拼接操作

##### `torch.cat(tensors, dim)`

沿着指定维度拼接多个张量。

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)  # 在第0维拼接
print(concatenated_tensor)
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])
```

##### `torch.stack(tensors, dim)`

在新维度上堆叠多个张量。

```python
import torch

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
stacked_tensor = torch.stack((tensor1, tensor2), dim=1)  # 在第1维堆叠
print(stacked_tensor)
# 输出:
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
```

### 5. 张量变换

#### 5.1 重塑形状

##### `tensor.view(shape)`

返回给定形状的张量视图，原始张量的形状必须与新形状兼容。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
reshaped_tensor = tensor.view(1, 4)  # 重塑为1x4
print(reshaped_tensor)
# 输出: tensor([[1, 2, 3, 4]])
```

##### `tensor.reshape(shape)`

改变张量的形状，返回一个具有指定形状的新张量。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
reshaped_tensor = tensor.reshape(1, 4)  # 重塑为1x4
print(reshaped_tensor)
# 输出: tensor([[1, 2, 3, 4]])
```

#### 5.2 转置与交换维度

##### `tensor.transpose(dim0, dim1)`

交换张量中两个维度的位置。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
transposed_tensor = tensor.transpose(0, 1)  # 交换第0维和第1维
print(transposed_tensor)
# 输出:
# tensor([[1, 3],
#         [2, 4]])
```

##### `tensor.permute(*dims)`

按照指定顺序排列张量的维度。

```python
import torch

tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
permuted_tensor = tensor.permute(1, 0, 2)  # 维度顺序变为(1, 0, 2)
print(permuted_tensor.shape)
# 输出: torch.Size([2, 2, 2])
```

#### 5.3 增减维度

##### `tensor.squeeze()`

删除所有长度为1的维度。

```python
import torch

tensor = torch.tensor([[[1, 2], [3, 4]]])
squeezed_tensor = tensor.squeeze()  # 删除多余的维度
print(squeezed_tensor)
# 输出:
# tensor([[1, 2],
#         [3, 4]])
```

##### `tensor.unsqueeze(dim)`

在指定位置增加一个长度为1的新维度。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
unsqueezed_tensor = tensor.unsqueeze(0)  # 在第0维增加一个新维度
print(unsqueezed_tensor)
# 输出:
# tensor([[[1, 2],
#          [3, 4]]])
```

### 6. 数学运算

#### 6.1 基本运算

##### `torch.add(x, y)`

对两个张量进行逐元素加法运算。

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.add(x, y)
print(result)
# 输出: tensor([5, 7, 9])
```

##### `torch.sub(x, y)`

对两个张量进行逐元素减法运算。

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.sub(x, y)
print(result)
# 输出: tensor([-3, -3, -3])
```

##### `torch.mul(x, y)`

对两个张量进行逐元素乘法运算。

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
result = torch.mul(x, y)
print(result)
# 输出: tensor([ 4, 10, 18])
```

##### `torch.div(x, y)`

对两个张量进行逐元素除法运算。

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
result = torch.div(x, y)
print(result)
# 输出: tensor([0.2500, 0.4000, 0.5000])
```

##### `torch.matmul(x, y)`

计算两个张量的矩阵乘法。

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(x, y)
print(result)
# 输出:
# tensor([[19, 22],
#         [43, 50]])
```

##### `torch.pow(base, exponent)`

计算张量的幂。

```python
import torch

base = torch.tensor([1, 2, 3])
exponent = 2
result = torch.pow(base, exponent)
print(result)
# 输出: tensor([1, 4, 9])
```

##### `torch.exp(tensor)`

计算张量中所有元素的指数。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
result = torch.exp(tensor)
print(result)
# 输出: tensor([  2.7183,   7.3891,  20.0855])
```

##### `torch.sqrt(tensor)`

计算张量中所有元素的平方根。

```python
import torch

tensor = torch.tensor([1.0, 4.0, 9.0])
result = torch.sqrt(tensor)
print(result)
# 输出: tensor([1.0000, 2.0000, 3.0000])
```

#### 6.2 汇总统计

##### `torch.sum(input)`

计算张量中所有元素的和。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
result = torch.sum(tensor)
print(result)
# 输出: tensor(10)
```

##### `torch.mean(input)`

计算张量中所有元素的平均值。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
result = torch.mean(tensor)
print(result)
# 输出: tensor(2.5000)
```

##### `torch.max(input)`

找出张量中所有元素的最大值。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
result = torch.max(tensor)
print(result)
# 输出: tensor(4)
```

##### `torch.min(input)`

找出张量中所有元素的最小值。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])
result = torch.min(tensor)
print(result)
# 输出: tensor(1)
```

##### `torch.std(input)`

计算张量中所有元素的标准差。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
result = torch.std(tensor)
print(result)
# 输出: tensor(1.29099)
```

##### `torch.var(input)`

计算张量中所有元素的方差。

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
result = torch.var(tensor)
print(result)
# 输出: tensor(1.6667)
```

### 7. 梯度相关操作

#### 7.1 标记张量需要计算梯度

使用 `tensor.requires_grad_()` 标记张量以便在反向传播中计算梯度。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
tensor.requires_grad_()
print(tensor)
# 输出: tensor([1., 2., 3.], requires_grad=True)
```

#### 7.2 获取张量的梯度

在进行反向传播后，`tensor.grad` 会包含相对于该张量的梯度。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor.sum().backward()
print(tensor.grad)
# 输出: tensor([1., 1., 1.])
```

#### 7.3 计算梯度

使用 `tensor.backward()` 计算张量的梯度值，前提是该张量需要计算梯度。

```python
import torch

tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor.sum().backward()
print(tensor.grad)
# 输出: tensor([1., 1., 1.])
```

### 8. 数据管理

#### 8.1 将张量移动到指定设备

使用 `tensor.to(device)` 将张量移动到指定的设备，如 GPU。

```python
import torch

tensor = torch.tensor([1, 2, 3])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
print(tensor.device)
# 输出: cuda:0 或 cpu
```

#### 8.2 保存与加载张量

##### `torch.save(obj, f)`

将对象保存到文件中。

```python
import torch

tensor = torch.tensor([1, 2, 3])
torch.save(tensor, 'tensor.pt')  # 将张量保存到文件
```

##### `torch.load(f)`

从文件中加载对象。

```python
import torch

tensor = torch.load('tensor.pt')  # 从文件加载张量
print(tensor)
# 输出: tensor([1, 2, 3])
```

### 9. 实用技巧

#### 9.1 类型转换

将张量转换为不同的数据类型。

```python
import torch

float_tensor = torch.randn((2, 3), dtype=torch.float32)
int_tensor = float_tensor.to(torch.int32)
print(int_tensor)
# 输出示例:
# tensor([[0, 0, 0],
#         [0, 0, 0]], dtype=torch.int32)
```

#### 9.2 复制与克隆

使用 `clone()` 创建张量的独立副本。

```python
import torch

original = torch.tensor([1, 2, 3])
copy = original.clone()
print(copy)
# 输出: tensor([1, 2, 3])
```

### 10. 注意事项

- **叶子节点**：只有叶子节点（在计算图的起点）的张量才会保存梯度。如果对叶子节点进行操作生成的新张量，默认 `requires_grad=True`，但不会保存 `grad`，除非调用 `retain_grad()`。
  
- **就地操作**：某些就地操作（如 `x += 1`）可能会破坏计算图，导致 Autograd 无法正确计算梯度。建议避免在需要梯度的张量上进行就地操作。

  ```python
  # 推荐
  y = x + 1
  
  # 不推荐
  x += 1  # 可能导致梯度计算错误
  ```

| 函数                                   | 描述                                              | 示例用法                      |
|----------------------------------------|---------------------------------------------------|-------------------------------|
| `torch.tensor(data)`                   | 从数据创建张量                                    | `torch.tensor([1, 2, 3])`     |
| `torch.zeros(size)`                    | 创建指定大小的全零张量                            | `torch.zeros((2, 3))`         |
| `torch.ones(size)`                     | 创建指定大小的全一张量                            | `torch.ones((2, 3))`          |
| `torch.empty(size)`                    | 创建未初始化的张量                                | `torch.empty((2, 3))`         |
| `torch.randn(size)`                    | 创建指定大小的标准正态分布张量                    | `torch.randn((2, 3))`         |
| `torch.arange(start, end, step)`        | 创建一个范围内的一维张量，步长为 `step`            | `torch.arange(0, 5, 1)`        |
| `torch.linspace(start, end, steps)`     | 创建一个范围内均匀间隔的一维张量，步数为 `steps`   | `torch.linspace(0, 5, 5)`     |
| `tensor.view(shape)`                    | 重塑张量形状                                      | `tensor.view(1, 4)`            |
| `tensor.reshape(shape)`                 | 改变张量形状                                      | `tensor.reshape(1, 4)`         |
| `tensor.transpose(dim0, dim1)`          | 交换张量的两个维度                                | `tensor.transpose(0, 1)`       |
| `tensor.permute(*dims)`                  | 按指定顺序排列张量的维度                          | `tensor.permute(1, 0, 2)`       |
| `tensor.squeeze()`                      | 删除所有长度为1的维度                              | `tensor.squeeze()`             |
| `tensor.unsqueeze(dim)`                  | 在指定位置增加一个长度为1的新维度                  | `tensor.unsqueeze(0)`          |
| `torch.add(x, y)`                       | 逐元素加法                                        | `torch.add(x, y)`               |
| `torch.sub(x, y)`                       | 逐元素减法                                        | `torch.sub(x, y)`               |
| `torch.mul(x, y)`                       | 逐元素乘法                                        | `torch.mul(x, y)`               |
| `torch.div(x, y)`                       | 逐元素除法                                        | `torch.div(x, y)`               |
| `torch.matmul(x, y)`                    | 矩阵乘法                                          | `torch.matmul(x, y)`            |
| `torch.pow(base, exponent)`             | 计算张量的幂                                      | `torch.pow(base, exponent)`     |
| `torch.exp(tensor)`                     | 计算张量中所有元素的指数                          | `torch.exp(tensor)`             |
| `torch.sqrt(tensor)`                    | 计算张量中所有元素的平方根                        | `torch.sqrt(tensor)`            |
| `torch.sum(input)`                      | 计算张量中所有元素的和                            | `torch.sum(tensor)`             |
| `torch.mean(input)`                     | 计算张量中所有元素的平均值                        | `torch.mean(tensor)`            |
| `torch.max(input)`                      | 找出张量中所有元素的最大值                        | `torch.max(tensor)`             |
| `torch.min(input)`                      | 找出张量中所有元素的最小值                        | `torch.min(tensor)`             |
| `torch.std(input)`                      | 计算张量中所有元素的标准差                        | `torch.std(tensor)`             |
| `torch.var(input)`                      | 计算张量中所有元素的方差                          | `torch.var(tensor)`             |
| `tensor.requires_grad_()`               | 标记张量需要计算梯度                              | `tensor.requires_grad_()`       |
| `tensor.backward()`                     | 计算张量的梯度值                                  | `tensor.backward()`             |
| `tensor.to(device)`                     | 将张量移动到指定设备（如GPU）                      | `tensor.to('cuda')`             |
| `torch.save(obj, f)`                    | 将对象保存到文件中                                 | `torch.save(tensor, 'file.pt')` |
| `torch.load(f)`                         | 从文件中加载对象                                   | `torch.load('file.pt')`          |
| `clone()`                               | 创建张量的独立副本                                 | `copy = original.clone()`        |


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
