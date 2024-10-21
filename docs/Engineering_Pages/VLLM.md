

# 使用vllm部署deepseek-prover教程

## 1. Pytorch环境安装

### 1.1 下载代码

从GitHub下载代码：

```bash
git clone --recurse-submodules git@github.com:deepseek-ai/DeepSeek-Prover-V1.5.git
```

如果服务器无法从GitHub下载，可以使用：

```bash
wget 'https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/archive/refs/heads/main.zip'
unzip 'main.zip'
```

### 1.2 修改requirements.txt

修改`requirements.txt`内容如下：

```
torch==2.2.1
pytz==2022.1
easydict==1.13
transformers==4.40.1
numpy==1.26.4
pandas==1.4.3
tabulate==0.9.0
termcolor==2.4.0
accelerate==0.33.0
flash_attn==2.6.3
vllm
```

注意：
- flash_attn 不能在 python>3.12上安装
- vllm 不指定版本

## 2. 模型下载

```bash
wget 'https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh'
bash script.deb.sh 
apt-get install git-lfs 
git lfs install

git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-Prover-V1.5-RL.git
```

## 3. Huggingface环境设置

为解决国内访问huggingface的问题，设置镜像网站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

永久生效方法：

```bash
vim ~/.bashrc
# 在文件末尾添加：
export HF_ENDPOINT=https://hf-mirror.com
# 使更改立即生效：
source ~/.bashrc
```

## 4. 模型推理

### 4.1 模型输入

使用mini-batch处理：

```python
batch_size = 256

keys = list(data)
new_data = data.copy()

num_batches = math.ceil(len(keys) / batch_size)

for i in tqdm(range(num_batches)):
    batch_keys = keys[i * batch_size:(i + 1) * batch_size]
    model_inputs = []
    
    for k in batch_keys:
        # 处理输入数据
        # ...

    if len(model_inputs) == 0:
        continue
    
    model_outputs = model.generate(
        model_inputs,
        sampling_params,
        use_tqdm=False,
    )
```

### 4.2 模型输出

```python
for idx, k in enumerate(batch_keys):
    # 处理模型输出
    # ...

    if isinstance(data[k], dict):
        # 更新字典数据
        # ...
    else:
        # 创建新字典
        # ...

    new_data[k] = tmp
```

## 5. 多卡并行

使用LLM包加载模型，支持数据并行和张量并行：

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

model = LLM(
    model=model_name,
    max_num_batched_tokens=8192,
    seed=1,
    trust_remote_code=True,
    tensor_parallel_size=4,  # 张量并行大小
    pipeline_parallel_size=1  # 流水线并行大小
)
```

## 6. 服务器资源监测

### 6.1 CPU使用情况

使用`htop`命令：

```bash
htop
```

### 6.2 GPU使用情况

使用`nvtop`命令：

```bash
nvtop
```