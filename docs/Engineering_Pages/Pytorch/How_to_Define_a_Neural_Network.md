[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/AIDIY?style=social)](https://github.com/InuyashaYang/AIDIY)

## 三、定义神经网络

### 1. 神经网络简介

神经网络（Neural Network）是由大量相互连接的神经元（节点）组成的计算模型，广泛应用于图像识别、自然语言处理、语音识别等领域。PyTorch 提供了灵活且高效的工具来定义和训练神经网络。

### 2. 使用 `torch.nn.Module` 定义神经网络

在 PyTorch 中，定义神经网络的基本方法是通过继承 `torch.nn.Module` 类，并定义网络的层和前向传播（forward）过程。

#### 2.1 继承 `nn.Module`

所有自定义的神经网络模型都应该继承自 `nn.Module`。这使得模型能够利用 PyTorch 提供的丰富功能，如参数管理、层定义等。

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义网络层

    def forward(self, x):
        # 定义前向传播过程
        return x
```

#### 2.2 定义网络层

在 `__init__` 方法中定义网络的各个层，例如全连接层、卷积层等。

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 输入层到隐藏层
        self.relu = nn.ReLU()            # 激活函数
        self.fc2 = nn.Linear(128, 10)    # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

### 3. 常用层与激活函数

PyTorch 提供了丰富的层（Layer）和激活函数（Activation Function），以下是一些常用的组件：

#### 3.1 全连接层（线性层）

```python
nn.Linear(in_features, out_features, bias=True)
```

- `in_features`：输入特征的数量
- `out_features`：输出特征的数量
- `bias`：是否包含偏置项

#### 3.2 卷积层

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

- `in_channels`：输入的通道数
- `out_channels`：输出的通道数
- `kernel_size`：卷积核的大小
- `stride`：步长
- `padding`：填充

#### 3.3 池化层

```python
nn.MaxPool2d(kernel_size, stride=None, padding=0)
nn.AvgPool2d(kernel_size, stride=None, padding=0)
```

- `kernel_size`：池化窗口的大小
- `stride`：步长
- `padding`：填充

#### 3.4 激活函数

```python
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.LeakyReLU()
```

#### 3.5 归一化层

```python
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
```

- `num_features`：特征数

### 4. 前向传播（Forward）

定义神经网络的前向传播过程，即数据如何从输入层流向输出层。

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
```

### 5. 示例：定义一个简单的全连接神经网络

以下是一个完整的示例，展示如何定义、初始化并打印一个简单的全连接神经网络。

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()                         # 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 初始化模型
input_size = 784
hidden_size = 128
num_classes = 10
model = SimpleNet(input_size, hidden_size, num_classes)

# 打印模型结构
print(model)
```

#### 输出：

```
SimpleNet(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```

### 6. 使用 `nn.Sequential` 定义神经网络

`nn.Sequential` 提供了一种更简洁的方式来定义神经网络，特别适用于层按顺序排列的模型。

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

print(model)
```

#### 输出：

```
Sequential(
  (0): Linear(in_features=784, out_features=128, bias=True)
  (1): ReLU()
  (2): Linear(in_features=128, out_features=10, bias=True)
)
```

### 7. 模型参数初始化

合理的参数初始化可以加速模型的收敛，提高模型性能。PyTorch 提供了多种初始化方法，可以自定义初始化或使用内置初始化。

#### 7.1 使用内置初始化方法

```python
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)

model.apply(init_weights)
```

#### 7.2 自定义初始化方法

```python
def custom_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(custom_init)
```

### 8. 模型的扩展与自定义

除了使用 `nn.Module` 和 `nn.Sequential`，你还可以创建更复杂的模型结构，如多输入多输出模型、带有跳跃连接的模型等。

#### 8.1 多输入多输出模型

```python
class MultiInputNet(nn.Module):
    def __init__(self):
        super(MultiInputNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x1, x2):
        out1 = self.fc1(x1)
        out2 = self.fc2(x2)
        out = out1 + out2
        return out
```

#### 8.2 带有跳跃连接的模型

```python
class ResNetBlock(nn.Module):
    def __init__(self, in_features):
        super(ResNetBlock, self).__init__()
        self.fc = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out += x  # 跳跃连接
        return out
```

### 9. 模型常用函数总结表

| 函数/模块                   | 描述                                           | 示例用法                                |
|-----------------------------|------------------------------------------------|-----------------------------------------|
| `nn.Module`                 | 所有神经网络模块的基类                         | `class MyModel(nn.Module):`             |
| `nn.Linear(in, out)`        | 全连接层                                        | `nn.Linear(784, 128)`                   |
| `nn.Conv2d(in, out, k)`     | 二维卷积层                                      | `nn.Conv2d(3, 16, 3)`                    |
| `nn.MaxPool2d(k)`            | 最大池化层                                      | `nn.MaxPool2d(2)`                        |
| `nn.ReLU()`                  | ReLU 激活函数                                  | `nn.ReLU()`                              |
| `nn.Sigmoid()`               | Sigmoid 激活函数                               | `nn.Sigmoid()`                           |
| `nn.Tanh()`                  | Tanh 激活函数                                  | `nn.Tanh()`                              |
| `nn.BatchNorm1d(num_features)` | 一维批归一化                                    | `nn.BatchNorm1d(128)`                    |
| `nn.BatchNorm2d(num_features)` | 二维批归一化                                    | `nn.BatchNorm2d(16)`                     |
| `nn.Sequential(*args)`       | 按顺序包装多个模块                             | `nn.Sequential(nn.Linear(784,128), nn.ReLU())` |
| `model.parameters()`        | 获取模型的所有参数                             | `for param in model.parameters():`       |
| `model.to(device)`          | 将模型移动到指定设备（如 GPU）                  | `model.to('cuda')`                       |
| `model.train()`             | 将模型设为训练模式（启用 dropout、batchnorm）    | `model.train()`                          |
| `model.eval()`              | 将模型设为评估模式（禁用 dropout、batchnorm）    | `model.eval()`                           |
| `model.apply(fn)`           | 对模型的每一层应用函数 `fn`                     | `model.apply(init_weights)`              |

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
