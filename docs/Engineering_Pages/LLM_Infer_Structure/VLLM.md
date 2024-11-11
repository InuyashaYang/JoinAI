# 在vLLM中的LLM类
[vLLM的github仓库](https://github.com/vllm-project/vllm)
# LLM 类

```python
class vllm.LLM(model: str, tokenizer: str | None = None, tokenizer_mode: str = 'auto', 
               skip_tokenizer_init: bool = False, trust_remote_code: bool = False, 
               tensor_parallel_size: int = 1, dtype: str = 'auto', 
               quantization: str | None = None, revision: str | None = None, 
               tokenizer_revision: str | None = None, seed: int = 0, 
               gpu_memory_utilization: float = 0.9, swap_space: float = 4, 
               cpu_offload_gb: float = 0, enforce_eager: bool | None = None, 
               max_seq_len_to_capture: int = 8192, disable_custom_all_reduce: bool = False, 
               disable_async_output_proc: bool = False, 
               mm_processor_kwargs: Dict[str, Any] | None = None, 
               task: Literal['auto', 'generate', 'embedding'] = 'auto', 
               pooling_type: str | None = None, pooling_norm: bool | None = None, 
               pooling_softmax: bool | None = None, pooling_step_tag_id: int | None = None, 
               pooling_returned_token_ids: List[int] | None = None, **kwargs)
```

LLM类用于从给定提示和采样参数生成文本。

该类包含一个分词器、一个语言模型(可能分布在多个GPU上)以及为中间状态(即KV缓存)分配的GPU内存空间。给定一批提示和采样参数,该类使用智能批处理机制和高效的内存管理从模型生成文本。

## 参数

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| model | str | - | HuggingFace Transformers模型的名称或路径 |
| tokenizer | str \| None | None | HuggingFace Transformers分词器的名称或路径 |
| tokenizer_mode | str | 'auto' | 分词器模式。"auto"将在可用时使用快速分词器,"slow"将始终使用慢速分词器 |
| skip_tokenizer_init | bool | False | 如果为True,跳过分词器和解码器的初始化。期望输入中的prompt_token_ids有效,prompt为None |
| trust_remote_code | bool | False | 下载模型和分词器时是否信任远程代码(例如来自HuggingFace) |
| tensor_parallel_size | int | 1 | 用于张量并行分布式执行的GPU数量 |
| dtype | str | 'auto' | 模型权重和激活的数据类型。支持float32、float16和bfloat16 |
| quantization | str \| None | None | 用于量化模型权重的方法。支持"awq"、"gptq"和"fp8"(实验性) |
| revision | str \| None | None | 要使用的特定模型版本。可以是分支名、标签名或提交ID |
| tokenizer_revision | str \| None | None | 要使用的特定分词器版本。可以是分支名、标签名或提交ID |
| seed | int | 0 | 用于初始化采样随机数生成器的种子 |
| gpu_memory_utilization | float | 0.9 | 为模型权重、激活和KV缓存保留的GPU内存比率(0到1之间) |
| swap_space | float | 4 | 每个GPU用作交换空间的CPU内存大小(GiB) |
| cpu_offload_gb | float | 0 | 用于卸载模型权重的CPU内存大小(GiB) |
| enforce_eager | bool \| None | None | 是否强制执行急切执行 |
| max_seq_len_to_capture | int | 8192 | CUDA图覆盖的最大序列长度 |
| disable_custom_all_reduce | bool | False | 见ParallelConfig |
| **kwargs | - | - | EngineArgs的参数(见Engine Arguments) |

## 方法

### beam_search

```python
beam_search(prompts: List[str | List[int]], params: BeamSearchParams) → List[BeamSearchOutput]
```

使用束搜索生成序列。

参数:
- prompts: 提示列表。每个提示可以是字符串或token ID列表。
- params: 束搜索参数。

### chat

```python
chat(messages: List[ChatCompletionMessageParam] | List[List[ChatCompletionMessageParam]], 
     sampling_params: SamplingParams | List[SamplingParams] | None = None,
     use_tqdm: bool = True, 
     lora_request: LoRARequest | None = None,
     chat_template: str | None = None,
     add_generation_prompt: bool = True,
     continue_final_message: bool = False,
     tools: List[Dict[str, Any]] | None = None,
     mm_processor_kwargs: Dict[str, Any] | None = None) → List[RequestOutput]
```

为聊天对话生成响应。

参数:
- messages: 对话列表或单个对话。
- sampling_params: 文本生成的采样参数。
- use_tqdm: 是否使用tqdm显示进度条。
- lora_request: 用于生成的LoRA请求(如果有)。
- chat_template: 用于构建聊天的模板。
- add_generation_prompt: 是否为每条消息添加生成模板。
- continue_final_message: 是否继续对话的最后一条消息。
- mm_processor_kwargs: 多模态处理器的kwargs覆盖。

返回:
包含生成的响应的RequestOutput对象列表。

### encode

```python
encode(prompts: PromptType | Sequence[PromptType], /, 
       *, pooling_params: PoolingParams | Sequence[PoolingParams] | None = None,
       use_tqdm: bool = True,
       lora_request: List[LoRARequest] | LoRARequest | None = None) → List[EmbeddingRequestOutput]
```

生成输入提示的嵌入。

参数:
- prompts: 提供给LLM的提示。
- pooling_params: 池化参数。
- use_tqdm: 是否使用tqdm显示进度条。
- lora_request: 用于生成的LoRA请求(如果有)。

返回:
包含生成的嵌入的EmbeddingRequestOutput对象列表。

### generate

```python
generate(prompts: PromptType | Sequence[PromptType], /, 
         *, sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
         use_tqdm: bool = True,
         lora_request: List[LoRARequest] | LoRARequest | None = None) → List[RequestOutput]
```

生成输入提示的补全。

参数:
- prompts: 提供给LLM的提示。
- sampling_params: 文本生成的采样参数。
- use_tqdm: 是否使用tqdm显示进度条。
- lora_request: 用于生成的LoRA请求(如果有)。

返回:
包含生成的补全的RequestOutput对象列表。

## 注意事项

- 此类旨在用于离线推理。对于在线服务,请使用AsyncLLMEngine类。
- 使用prompts和prompt_token_ids作为关键字参数被视为遗留用法,可能在未来被弃用。您应该通过inputs参数传递它们。