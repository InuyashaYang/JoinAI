# LlamaFactory的安装与使用(特别在lean环境中)

# 一、安装

具体安装过程参考：https://zhuanlan.zhihu.com/p/695287607

1、从git上拉取代码

```
git clone https://github.com/hiyouga/LLaMA-Factory.git

```

2、在对应的环境里安装

```
cd LLaMA-Factory
pip install -e '.[torch,metrics]'

```

3、校验安装的结果

```
import torch
torch.cuda.current_device()
torch.cuda.get_device_name(0)
torch.version

```

```
llamafactory-cli train -h

```

> 在安装的过程中遇到问题：
> 
> 
> 安装llamafactory-cli以及nltk等一系列command时，遇到warning，WARNING: The script isympy is installed in '/root/.local/bin' which is not on PATH.
> 
> Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
> 
> 解决办法：
> 
> 需要将对应的路径添加到环境变量里面，才能使用 llamafactory-cli 命令
> 
> export PATH="$PATH:/root/.local/bin"
> 
> source ~/.bashrc
> 

如果以上指令都可以正确的运行，那么Llamafactory就安装完毕了

# 二、推理

直接加载下载好的 Llama 模型做 infer

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /root/dataDisk/Meta-Llama-3-8B-Instruct \
    --template llama3

```

这里会存在一个小问题。运行以上代码后，会默认在本地的 7860 端口 （gradio的默认端口号） run图形化界面。对于一些服务器，会要求指定的端口，如潞晨云是6006。llamafactory-cli 貌似没有提供修改的端口的参数？？

因此，可以通过以下指令直接修改gradio的默认端口号，即可在6006端口运行图形化界面

```
export GRADIO_SERVER_PORT=6006
source ~/.bashrc

```

最后在服务器打开图形化界面进行推理

# 三、数据集的构建

系统目前支持 [alpaca](https://zhida.zhihu.com/search?content_id=242638741&content_type=Article&match_order=1&q=alpaca&zhida_source=entity) 和 sharegpt 两种数据格式，以alpaca单轮的sft数据为例，要求格式为：

```
{
  "instruction": "写一个有效的比较语句",
  "input": "篮球和足球",
  "output": "篮球和足球都是受欢迎的运动。"
}

```

数据集已经是该格式，将数据集文件命名为：lean4_v1_1009.json，放在 ./LLaMA-Factory-main/data/lean4_v1_1009.json 路径下，并修改该路径下的 dataset_info.json 文件，添加数据集lean4

# 四、微调

在8张A800上进行训练：

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /root/dataDisk/Meta-Llama-3-8B-Instruct \
    --dataset lean4 \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_typelora\
    --output_dir ./saves/LLaMA3-8B-Instruct/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 18 \
    --per_device_eval_batch_size 18 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 50 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --plot_loss \
    --fp16

```

# 五、评估(lean4 Minif2f)

把miniF2F处理成如下格式，进行评估

```
{
        "output": "theorem mathd_algebra_478\n  (b h v : ℝ)\n  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)\n  (h₁ : v = 1 / 3 * (b * h))\n  (h₂ : b = 30)\n  (h₃ : h = 13 / 2) :\n  v = 65 := sorry",
        "input": "The volume of a cone is given by the formula $V = \\frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.",
        "instruction": "You are an expert in Lean4 theorem prover and you will be given a theorem in natural language, and you must translate it into the correct formal statement in Lean4."
}

```

批量评估指令：

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /root/dataDisk/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ./saves/LLaMA3-8B-Instruct/lora/sft  \
    --eval_dataset miniF2F \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type lora \
    --output_dir ./saves/Meta-Llama-3-8B-Instruct/lora/predict \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 20 \
    --predict_with_generate

```

合并lora的ui界面

```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /root/dataDisk/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path ./saves/LLaMA3-8B-Instruct/lora/sft  \
    --template llama3 \
    --finetuning_type lora

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
