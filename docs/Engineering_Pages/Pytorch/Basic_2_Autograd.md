[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/AIDIY?style=social)](https://github.com/InuyashaYang/AIDIY)

## 二、自动求导（Autograd）

### 1. 什么是 Autograd？

Autograd 是 PyTorch 的自动微分系统，能够根据计算图自动计算张量的梯度。这在训练神经网络时尤为重要，因为需要反向传播算法来更新模型参数。

### 2. 计算图与自动微分

每当你对张量执行一个操作时，PyTorch 都会创建一个计算图，该图记录了这些操作的顺序和关系。Autograd 利用这个图来自动计算梯度。

### 3. 关键概念与属性

- **requires_grad**

  设置张量的 `requires_grad` 属性为 `True`，表示需要对该张量进行梯度计算。

  ```python
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  ```

- **grad**

  在进行反向传播后，`x.grad` 会包含相对于 `x` 的梯度。

  ```python
  y = x * 2
  y = y.sum()
  y.backward()
  print(x.grad)  # 输出: tensor([2., 2., 2.])
  ```

### 4. 基本用法

#### 4.1 简单的梯度计算

```python
import torch

# 创建一个需要梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 定义一个简单的函数
y = x * 2
z = y.sum()

# 反向传播
z.backward()

# 查看梯度
print(x.grad)  # 输出: tensor([2., 2., 2.])
```

#### 4.2 多步计算

```python
import torch

x = torch.randn(3, requires_grad=True)
y = x * 3
z = y ** 2
out = z.mean()

out.backward()
print(x.grad)
```

### 5. 高级用法

#### 5.1 不需要梯度的操作

在某些情况下，你可能不希望计算梯度，可以使用 `torch.no_grad()` 或 `detach()`。

- **torch.no_grad()**

  ```python
  with torch.no_grad():
      y = x * 2
  ```

- **detach()**

  ```python
  y = x.detach()
  ```

#### 5.2 创建不可求导的张量

有时需要临时禁用梯度计算，可以使用 `requires_grad_()`。

```python
x = torch.randn(3, requires_grad=True)
x.requires_grad_(False)
```

### 6. 处理梯度

#### 6.1 清零梯度

在每次反向传播前，通常需要清零梯度，以避免梯度累积。

```python
optimizer.zero_grad()
```

#### 6.2 访问梯度

梯度存储在 `grad` 属性中。

```python
print(x.grad)
```

#### 6.3 梯度累积

默认情况下，PyTorch 会累积梯度。这在某些优化策略中可能有用，但通常需要手动清零。

### 7. 应用场景

- **训练神经网络**

  在训练过程中，使用 Autograd 计算损失函数关于模型参数的梯度，以便使用优化器更新参数。

- **自定义损失函数**

  可以定义任意复杂的损失函数，Autograd 会自动处理梯度计算。

### 8. 注意事项

- **叶子节点**

  只有叶子节点（在计算图的起点）的张量才会保存梯度。如果对叶子节点进行操作，生成的新张量默认 `requires_grad=True`，但 `grad` 不会被保存，除非设置 `retain_grad()`。

  ```python
  x = torch.randn(3, requires_grad=True)
  y = x * 2
  y.requires_grad_(True)
  z = y.sum()
  z.backward()
  print(x.grad)  # 正常获取梯度
  print(y.grad)  # None unless y.retain_grad() is called before backward
  ```

- **就地操作**

  某些就地操作（如 `x += 1`）可能会破坏计算图，导致 Autograd 无法正确计算梯度。建议尽量避免在需要梯度的张量上进行就地操作。

  ```python
  # 推荐
  y = x + 1
  
  # 不推荐
  x += 1  # 可能导致梯度计算错误
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
