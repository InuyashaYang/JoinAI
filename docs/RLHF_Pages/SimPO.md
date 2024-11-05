
## SimPO vs DPO

### DPO的Loss回顾
DPO的loss是：
$\mathcal{L}_\text{DPO} = -\log(\sigma(r(x,y_w,y_l)))$

### SimPO的Loss
SimPO直接使用MSE (Mean Squared Error) loss：
$\mathcal{L}_\text{SimPO} = (r(x,y_w,y_l) - \beta)^2$

这里：
- $r(x,y_w,y_l)$ 的计算方式与DPO相同
- $\beta$ 是一个超参数，通常设置为正数（如1或2）
- 这个loss函数的目标是让偏好分数 $r$ 接近目标值 $\beta$

## SimPO的优势
1. 计算更简单：不需要sigmoid和对数计算
2. 训练更稳定：MSE loss比DPO的交叉熵loss数值更稳定
3. 通过调节 $\beta$ 可以直接控制想要的偏好强度

## 共同点
1. 都使用reference model作为基准
2. 都是自回归式计算序列概率
3. 都不需要显式的奖励函数
4. 训练过程都只更新policy model参数

SimPO是DPO的一个更简单的替代方案，特别适合在计算资源有限或需要更稳定训练过程的场景。