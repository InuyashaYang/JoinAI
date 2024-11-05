 # DPO (Direct Preference Optimization) 训练流程

## 1. 输入数据格式

- 问题序列: $x$
- 较好的回答序列: $y_w$
- 较差的回答序列: $y_l$

## 2. 前向传播过程

### 2.1 对较好回答 $y_w$ 的概率计算
1. Policy Model计算:
   $\log \pi_\theta(y_w|x) = \sum_{t=1}^T \log \pi_\theta(y_w^t|x,y_w^{<t})$

2. Reference Model计算(参数固定):
   $\log \pi_\text{ref}(y_w|x) = \sum_{t=1}^T \log \pi_\text{ref}(y_w^t|x,y_w^{<t})$

### 2.2 对较差回答 $y_l$ 的概率计算
1. Policy Model计算:
   $\log \pi_\theta(y_l|x) = \sum_{t=1}^T \log \pi_\theta(y_l^t|x,y_l^{<t})$

2. Reference Model计算(参数固定):
   $\log \pi_\text{ref}(y_l|x) = \sum_{t=1}^T \log \pi_\text{ref}(y_l^t|x,y_l^{<t})$

### 2.3 计算偏好分数
$r(x,y_w,y_l) = \log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}$

等价于:
$r(x,y_w,y_l) = [\log \pi_\theta(y_w|x) - \log \pi_\text{ref}(y_w|x)] - [\log \pi_\theta(y_l|x) - \log \pi_\text{ref}(y_l|x)]$

## 3. Loss计算
$\mathcal{L}_\text{DPO} = -\log(\sigma(r(x,y_w,y_l)))$

其中 $\sigma$ 是sigmoid函数。

## 4. 训练要点
1. 只更新policy model的参数 $\theta$
2. Reference model的参数在训练过程中保持固定
3. 每个token都是自回归生成的条件概率
4. 通过最小化loss来增大policy model对好回答的偏好程度

这个过程的关键在于:

- 自回归方式计算序列概率
- 使用reference model作为正则化基准
- 直接利用人类偏好数据,无需显式奖励函数