```
GPU
|-- GPC (Graphics Processing Cluster)
    |-- TPC (Texture Processing Cluster)
        |-- SM (Streaming Multiprocessor)
            |-- Warp (执行单位，不是硬件)
            |-- SP (Stream Processor) / CUDA Core
                |-- ALU (Arithmetic Logic Unit)
                |-- FPU (Floating Point Unit)
            |-- SFU (Special Function Unit)
            |-- Load/Store Unit
            |-- Register File
            |-- L1 Cache
|-- ROP (Render Output Unit)
|-- L2 Cache
|-- Memory Controller
    |-- Memory (GDDR6, HBM2 等)
```
# GPU 组件分类

| ALU (算术逻辑单元) | Cache (缓存) | Control (控制单元) |
|-------------------|-------------|-------------------|
| CUDA Cores / Stream Processors | L1 缓存 | 调度器 |
| 浮点单元 (FPU) | L2 缓存 | 指令分发单元 |
| 特殊函数单元 (SFU) | 共享内存 | 内存控制器 |
| Tensor Cores (1) | 寄存器文件 | PCIe 接口控制器 |
| RT Cores (1) | 纹理缓存 | Warp 调度器 |
| 整数单元 | 常量缓存 | GPC 控制逻辑 |
| 纹理单元 (2) | | TPC 控制逻辑 |
| ROP (渲染输出单元) (2) | | SM 控制逻辑 |
| | | 负载均衡器 |

## 注释

(1) Tensor Cores 和 RT Cores 仅存在于较新的 NVIDIA GPU 架构中。

(2) 纹理单元和 ROP 包含计算和缓存功能，但主要归类为 ALU。

## 附加说明

1. 内存子系统（如 GDDR6 或 HBM2）通常被视为独立于这三个类别，但与所有类别都有密切的交互。
2. 某些控制功能可能集成在其他单元中，如 SM 或 GPC 内的控制逻辑。
3. 不同 GPU 架构可能有额外的专用单元或略有不同的组织方式。
4. 这个分类是一个简化模型，实际 GPU 架构可能更加复杂和集成。

# GPU ALU (算术逻辑单元) 组件分类

## 1. 核心计算单元
- CUDA Cores / Stream Processors
  - 整数单元 (INT)
  - 浮点单元 (FPU)
    - 单精度浮点 (FP32)
    - 双精度浮点 (FP64)
    - 半精度浮点 (FP16)

## 2. 专用计算单元
- 特殊函数单元 (SFU)
  - 三角函数
  - 指数函数
  - 对数函数
- Tensor Cores (1)
  - 矩阵乘法
  - AI 加速
- RT Cores (1)
  - 光线追踪加速

## 3. 图形处理单元
- 纹理单元 (2)
  - 纹理过滤
  - 纹理寻址
- ROP (渲染输出单元) (2)
  - 像素操作
  - 颜色混合

## 注释
(1) Tensor Cores 和 RT Cores 仅存在于较新的 NVIDIA GPU 架构中。

(2) 纹理单元和 ROP 虽然也包含缓存和控制功能，但在计算方面主要归类为 ALU。

## 附加说明
1. 不同 GPU 架构可能会有不同的 ALU 组件配置和数量。
2. 某些新型 GPU 可能包含额外的专用计算单元，如用于 AI 推理的单元。
3. ALU 的具体实现和性能特性可能因 GPU 型号和代际而异。
4. 部分 ALU 功能可能与缓存和控制单元紧密集成，边界并非总是明确。


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
