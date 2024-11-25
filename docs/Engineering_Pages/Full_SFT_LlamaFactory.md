
# 使用LLamaFactory进行全量微调

## 一、实现流程记录

### 1. 环境配置

在已经安装好LLamaFactory的基础上，还需要安装DeepSpeed来解决CUDA显存问题：

```bash
pip install deepspeed
```

安装完成后，在待运行脚本的目录下创建`ds_config.json`文件：

```json
{
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_fp16_weights_on_model_save": false
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### 2. 全量微调

运行LLamaFactory的微调脚本：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /root/dataDisk/Meta-Llama-3-8B-Instruct \
    --dataset lean4_v2,lean_workbook \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type full \
    --output_dir ./saves/LLaMA3-8B-Instruct/full/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 50 \
    --eval_steps 100 \
    --save_steps 400 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    --deepspeed ds_config.json
```

注意：微调过程中会存储模型参数、优化器参数等，需要的空间远比模型的大小要大得多，建议准备几百G或T级别的存储空间。

### 3. 推理

微调后的checkpoint文件可能无法直接加载，需要进行额外步骤获取完整的模型参数：

1. 在checkpoint文件夹中运行`zero_to_fp32.py`脚本：

```bash
python zero_to_fp32.py ./ pytorch_model.bin
```

2. 使用以下脚本进行推理：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path xxx/checkpoint-xxx \
    --eval_dataset miniF2F \
    --dataset_dir ./data \
    --template llama3 \
    --output_dir ./saves/xxx/miniF2F \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 20 \
    --predict_with_generate
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
