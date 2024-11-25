
## **MCTS-DPO 方法概述**

**MCTS-DPO** 是一种结合蒙特卡洛树搜索（MCTS）与直接偏好优化（DPO）的迭代方法，旨在通过逐步优化策略模型（如大型语言模型，LLM）的偏好学习过程，从而提升其推理能力和响应质量。

---

## **1. 输入与初始化**

- **初始策略**：$\pi_{\theta^{(0)}}$，通常为预训练或初步微调后的模型策略。
- **提示数据集**：$\mathcal{D}_{\mathcal{P}}$，包含多个输入提示（prompts）。
- **其他参数**：
  - $M$：迭代次数。
  - $B$：每次迭代中采样的提示数量。
  - $T$：每个推理链的最大步骤数（深度）。
  - $K$：每步的MCTS迭代次数。
  - $c_{\mathrm{puct}}$ 和 $\lambda$：用于PUCT公式的超参数。
  - $b_1$ 和 $b_2$：搜索树初始和后续的扩展幅度。

---

## **2. 迭代流程**

整个流程包括多个迭代，每次迭代均包括数据采样、MCTS搜索、偏好数据提取以及策略更新。具体步骤如下：

### **2.1 数据采样**

在第 $i$ 次迭代中，从提示数据集 $\mathcal{D}_{\mathcal{P}}$ 中随机采样 $B$ 个提示，形成当前迭代的子集：

$$
\mathcal{D}_{\mathcal{P}}^{(i)} \subseteq \mathcal{D}_{\mathcal{P}}
$$

### **2.2 MCTS 搜索与偏好数据提取**

对于每个提示 $x \in \mathcal{D}_{\mathcal{P}}^{(i)}$，使用当前策略 $\pi_{\theta^{(i-1)}}$ 通过蒙特卡洛树搜索（MCTS）构建一个深度为 $T$ 的搜索树：

1. **状态定义**：
   - 每个步骤的状态 $s_t$ 定义为当前推理链的前缀。
   - 执行动作 $a$ 后，状态转移至 $s_{t+1}$，即：

     $s_{t+1} = s_t + a$

2. **MCTS 过程**：
   - **选择（Select）**：利用PUCT策略选择下一个要扩展的节点，平衡探索与利用：

     
     ${s_{t+1}^*} = \arg\max_{a} \left[ Q(s_t, a) + c_{\mathrm{puct}} \cdot p(a \mid s_t) \frac{\sqrt{N(s_t)}}{1 + N(s_{t+1}^*)} \right]$
    

     其中，$p(a \mid s_t) = \frac{\pi_{\theta}(a \mid x, s_t)}{|a|^{\lambda}}$。

   - **扩展（Expand）**：在叶节点处生成新的动作，计算其奖励：

     $R(s_t) = \mathcal{O}(s_t) + \mathcal{C}(s_t)$

     其中，$\mathcal{O}(s_t)$ 表示结果正确性，$\mathcal{C}(s_t)$ 表示自我评估。

   - **备份（Backup）**：将评估结果向上传播，更新 $Q$ 值和访问次数：

    $Q(s_t, a) \leftarrow R(s_t, a) + \gamma V(s_{t+1})$

    $V(s_t) \leftarrow \frac{\sum_{a} N(s_{t+1}) Q(s_t, a)}{\sum_{a} N(s_{t+1})}$

    $N(s_t) \leftarrow N(s_t) + 1$

3. **偏好数据提取**：
   - 对于搜索树的每个深度 $t$，选择具有最高 $Q$ 值的步骤 $y_w^{(j,t)}$ 作为**正样本**，以及具有最低 $Q$ 值的步骤 $y_l^{(j,t)}$ 作为**负样本**，形成偏好对：
     
     $\mathcal{D}_i = \left\{(x^j, y_w^{(j,t)}, y_l^{(j,t)}) \mid x^j \sim \mathcal{D}_{\mathcal{P}}^{(i)}, t = 1, \ldots, T \right\}$

### **2.3 策略更新（DPO 优化）**

利用提取的偏好数据 $\mathcal{D}_i$，通过直接偏好优化（DPO）方法更新策略参数 $\theta$：

1. **损失函数定义**：

   
   $\ell_{i}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_i} \left[(1 - \alpha_{x, y_w, y_l}) \log \sigma (\beta h_{\pi_{\theta}}^{y_w, y_l}) + \alpha_{x, y_w, y_l} \log \sigma (-\beta h_{\pi_{\theta}}^{y_w, y_l}) \right]$

   其中，

   $$
   h_{\pi_{\theta}}^{y_w, y_l} = \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{r}}(y_w \mid x)} - \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{r}}(y_l \mid x)}
   $$

   $$
   \alpha_{x, y_w, y_l} = \frac{1}{\frac{N(x, y_w)}{N(x, y_l)} + 1}
   $$

2. **参数优化**：
   通过最小化损失函数 $\ell_i(\theta)$ 更新模型参数 $\theta$，得到新的策略 $\pi_{\theta^{(i)}}$：

   $$
   \theta \leftarrow \theta - \eta \nabla_\theta \ell_i(\theta)
   $$

### **2.4 迭代循环**

重复上述数据采样、MCTS 搜索、偏好数据提取及策略更新的过程，共进行 $M$ 次迭代。每次迭代后，更新后的策略用于下一轮的MCTS数据采集，确保模型在每轮迭代中不断优化和自我提升。

---

## **3. 具体算法流程**

以下是 **MCTS-DPO** 的具体算法流程，使用数学符号进行描述：

### **算法步骤**

1. **输入**：
   - $\mathcal{D}_{\mathcal{P}}$：提示数据集。
   - $q(\cdot \mid x)$：MCTS 采样策略，基于策略 $\pi$ 进行响应生成与自我评估。
   - $\ell_i(x, y_w, y_l; \theta)$：第 $i$ 次迭代的偏好学习损失函数。
   - $M$：迭代次数。
   - $B$：每次迭代的样本数量。
   - $T$：每个样本的平均步骤数。

2. **初始训练**：
   - 使用步骤级别的偏好学习在提示数据集 $\mathcal{D}_{\mathcal{P}}$ 上训练初始策略 $\pi_{\theta}$。

3. **迭代过程**（对 $i = 1$ 到 $M$）：
   - 设置当前策略 $\pi^{(i)} \leftarrow \pi_{\theta^{(i-1)}}$。
   - 从 $\mathcal{D}_{\mathcal{P}}$ 中采样 $B$ 个提示，形成 $\mathcal{D}_{\mathcal{P}}^{(i)}$。
   - **MCTS 搜索**：
     - 对每个 $x \in \mathcal{D}_{\mathcal{P}}^{(i)}$，使用 $q_{\pi_{\theta}}(\cdot \mid x)$ 构建深度为 $T$ 的搜索树。
     - 提取偏好数据 $\mathcal{D}_i$，包含每个深度 $t$ 的最优和最差步骤对 $(y_w^{(j,t)}, y_l^{(j,t)})$。
   - **DPO 优化**：
     - 使用 $\mathcal{D}_i$ 优化参数 $\theta$，最小化损失函数：

       $$
       J(\theta) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_i} \ell_i(x, y_w, y_l; \theta)
       $$

     - 更新策略 $\pi_{\theta^{(i)}}$。

4. **输出**：
   - 经过 $M$ 次迭代后的最终策略 $\pi_{\theta}$。

---

## **4. 关键数学公式解释**

- **PUCT 选择策略**：

  $$
  {s_{t+1}^*} = \arg\max_{a} \left[ Q(s_t, a) + c_{\mathrm{puct}} \cdot p(a \mid s_t) \frac{\sqrt{N(s_t)}}{1 + N(s_{t+1}^*)} \right]
  $$

  用于在选择阶段平衡探索与利用。

- **奖励计算**：

  $$
  R(s_t) = \mathcal{O}(s_t) + \mathcal{C}(s_t)
  $$

  其中，$\mathcal{O}(s_t)$ 是结果正确性，$\mathcal{C}(s_t)$ 是自我评估的信心分数。

- **动作价值更新**：

  $$
  Q(s_t, a) \leftarrow R(s_t, a) + \gamma V(s_{t+1})
  $$

- **状态值更新**：

  $$
  V(s_t) \leftarrow \frac{\sum_{a} N(s_{t+1}) Q(s_t, a)}{\sum_{a} N(s_{t+1})}
  $$

- **访问次数更新**：

  $$
  N(s_t) \leftarrow N(s_t) + 1
  $$

- **DPO 损失函数**：

  $$
  \ell_{i}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_i} \left[(1 - \alpha_{x, y_w, y_l}) \log \sigma (\beta h_{\pi_{\theta}}^{y_w, y_l}) + \alpha_{x, y_w, y_l} \log \sigma (-\beta h_{\pi_{\theta}}^{y_w, y_l}) \right]
  $$

  其中，

  $$
  h_{\pi_{\theta}}^{y_w, y_l} = \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{r}}(y_w \mid x)} - \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{r}}(y_l \mid x)}
  $$

  $$
  \alpha_{x, y_w, y_l} = \frac{1}{\frac{N(x, y_w)}{N(x, y_l)} + 1}
  $$
