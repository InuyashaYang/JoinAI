[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/AIDIY?style=social)](https://github.com/InuyashaYang/AIDIY)

## 目录
1. **训练循环概述**
2. **准备工作**
   - 模型实例化
   - 选择设备（CPU/GPU）
   - 定义损失函数和优化器
3. **构建训练循环**
   - 训练过程
   - 验证过程
4. **完整示例**
5. **实用技巧与注意事项**
6. **常见问题解答**

---

## 一、训练循环概述

**训练循环** 是指在训练过程中，模型通过多次迭代（epoch）学习数据，不断调整参数以最小化损失函数。一个典型的训练循环包括以下步骤：

1. **数据加载**：通过 `DataLoader` 获取训练和验证数据批次。
2. **前向传播（Forward Pass）**：将输入数据通过模型，得到预测输出。
3. **计算损失**：比较预测输出与真实标签，计算损失值。
4. **反向传播（Backward Pass）**：计算损失函数相对于模型参数的梯度。
5. **优化参数**：使用优化器根据梯度更新模型参数。
6. **验证模型**（可选）：在验证集上评估模型性能，监控过拟合情况。
7. **记录与监控**：记录训练过程中的损失和准确率，便于分析和调优。

---

## 二、准备工作

在开始构建训练循环之前，需要完成一些准备工作，包括实例化模型、选择设备、定义损失函数和优化器。

### 1. 模型实例化

假设我们使用之前定义的 `SimpleNet` 模型：

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()                         # 激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 实例化模型
model = SimpleNet()
print(model)
```

### 2. 选择设备（CPU/GPU）

为了加速模型训练，通常会使用 GPU（如果可用）。以下代码检查是否有可用的 GPU，并将模型移动到相应设备上。

```python
# 检查 GPU 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 将模型移动到设备上
model.to(device)
```

### 3. 定义损失函数和优化器

选择合适的损失函数和优化器对于模型的训练至关重要。

```python
# 定义损失函数（例如交叉熵损失）
criterion = nn.CrossEntropyLoss()

# 定义优化器（例如随机梯度下降 SGD）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

> **注**：除了 `SGD`，PyTorch 还提供了多种优化器，如 `Adam`、`RMSprop` 等。可以根据具体任务选择合适的优化器。

---

## 三、构建训练循环

### 1. 训练过程

以下是一个典型的训练循环结构：

```python
num_epochs = 10  # 训练的总轮数

for epoch in range(num_epochs):
    model.train()  # 将模型设为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 将数据移动到设备
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 若使用的是卷积神经网络，需要调整输入形状
        inputs = inputs.view(inputs.size(0), -1)  # 展平输入
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 统计损失
        running_loss += loss.item() * inputs.size(0)
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
```

### 2. 验证过程

在每个 epoch 结束后，可以在验证集或测试集上评估模型性能，以监控过拟合情况。

```python
    # 验证过程
    model.eval()  # 将模型设为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(test_dataset)
    val_epoch_acc = 100. * correct / total
    print(f'验证 Loss: {val_epoch_loss:.4f}, 准确率: {val_epoch_acc:.2f}%')
```

### 3. 完整训练和验证循环

将训练和验证过程结合起来：

```python
num_epochs = 10  # 训练的总轮数

for epoch in range(num_epochs):
    # 训练阶段
    model.train()  # 将模型设为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total
    
    # 验证阶段
    model.eval()  # 将模型设为评估模式
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(test_dataset)
    val_epoch_acc = 100. * correct_val / total_val
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
          f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')
```

---

## 四、完整示例

以下是一个结合前述所有步骤的完整训练循环示例，使用 MNIST 数据集和 `SimpleNet` 模型。

### 1. 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 超参数
input_size = 784  # 28x28
hidden_size = 128
num_classes = 10
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 实例化模型、定义损失函数和优化器
model = SimpleNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.view(inputs.size(0), -1)  # 展平
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_epoch_loss = val_loss / len(test_dataset)
    val_epoch_acc = 100. * correct_val / total_val
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
          f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')
```

### 2. 代码解释

- **数据加载**：使用 `torchvision.datasets.MNIST` 加载 MNIST 数据集，并通过 `DataLoader` 分批次加载数据。
- **模型实例化**：定义并实例化 `SimpleNet` 模型，并将其移动到选定设备（CPU 或 GPU）。
- **损失函数与优化器**：使用交叉熵损失函数 `nn.CrossEntropyLoss`，优化器选择 SGD。
- **训练循环**：
  - **训练阶段**：
    - 设置模型为训练模式 `model.train()`，启用如 Dropout 等层。
    - 迭代每个批次的数据，进行前向传播、计算损失、反向传播、参数更新。
    - 统计损失值和准确率。
  - **验证阶段**：
    - 设置模型为评估模式 `model.eval()`，禁用 Dropout 等层。
    - 禁用梯度计算 `with torch.no_grad()`，提高计算效率。
    - 迭代验证集数据，计算损失值和准确率。
  - **记录输出**：每个 epoch 结束后，输出训练和验证的损失值与准确率。

### 3. 输出示例

```
使用设备: cuda
Epoch [1/10], Train Loss: 0.4645, Train Acc: 83.21%, Val Loss: 0.1872, Val Acc: 94.83%
Epoch [2/10], Train Loss: 0.1609, Train Acc: 95.07%, Val Loss: 0.1403, Val Acc: 95.83%
...
Epoch [10/10], Train Loss: 0.0294, Train Acc: 99.57%, Val Loss: 0.0957, Val Acc: 96.64%
```

---

## 五、实用技巧与注意事项

### 1. **模型模式切换**

- **训练模式**：`model.train()` 启用训练模式，激活如 Dropout、BatchNorm 等层。
- **评估模式**：`model.eval()` 启用评估模式，禁用 Dropout、BatchNorm 等层。

### 2. **梯度清零**

在每次反向传播前，使用 `optimizer.zero_grad()` 清零梯度，以避免梯度累加。

### 3. **数据展平**

对于全连接网络，通常需要将输入图像展平为一维张量。例如，MNIST 图像 `28x28` 展平成 `784` 维。

### 4. **无梯度计算**

在验证阶段，使用 `torch.no_grad()` 上下文管理器，以节省计算资源和内存。

### 5. **保存与加载模型**

可以在训练过程中定期保存模型，以防训练中断或后续使用。

```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = SimpleNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
```

### 6. **学习率调度**

使用学习率调度器调整优化器的学习率，有助于模型更好地收敛。

```python
# 定义学习率调度器（每隔5个epoch将学习率减半）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(num_epochs):
    # 训练和验证过程
    ...
    # 更新学习率
    scheduler.step()
```

### 7. **记录与可视化**

使用工具如 TensorBoard、Matplotlib 等，记录训练和验证的损失与准确率，进行可视化分析。

```python
from torch.utils.tensorboard import SummaryWriter

# 初始化 TensorBoard writer
writer = SummaryWriter('runs/mnist_experiment')

for epoch in range(num_epochs):
    # 训练过程
    ...
    # 记录损失和准确率
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
    writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_epoch_acc, epoch)

writer.close()
```
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
