如何检查当前节点上的显卡数量
```python
import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # 打印每个 GPU 的名称
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")
```

安装方法：建议启用cuda12.1版本

```
#!/bin/bash

# 定义文件名
FILE="script.deb.sh"

# 检查 script.deb.sh 是否存在
if [ ! -f "$FILE" ]; then
    wget 'https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh'
    bash script.deb.sh
    apt-get install git-lfs  
    git lfs install
else
    echo "$FILE 已存在，跳过下载。"
fi

# 检查 DeepSeek-Prover 是否存在
if [ ! -d "/root/dataDisk/DeepSeek-Prover-V1.5-RL" ]; then
    git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-Prover-V1.5-RL.git /root/dataDisk/DeepSeek-Prover-V1.5-RL
else
    echo "DeepSeek-Prover-V1.5-RL 已存在，跳过克隆。"
fi

# 安装 Python 依赖
pip install torch==2.4.0;
pip install pytz==2022.1;
pip install easydict==1.13;
pip install transformers==4.40.1;
pip install numpy==1.26.4;
pip install pandas;
pip install tabulate==0.9.0;
pip install termcolor==2.4.0;
pip install accelerate==0.33.0;
pip install flash_attn==2.6.3;
pip install vllm;
```

一般性的模型加载
```python
import re
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "/root/dataDisk/DeepSeek-Prover-V1.5-RL"
tokenizer = AutoTokenizer.from_pretrained(model_name)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 加载模型
model = LLM(
    model=model_name,
    max_num_batched_tokens=8200,
    seed=1,
    trust_remote_code=True,
    tensor_parallel_size=8,  # 张量并行大小
    pipeline_parallel_size=1  # 流水线并行大小
)
```
