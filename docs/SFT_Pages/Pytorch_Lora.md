[![GitHub stars](https://img.shields.io/github/stars/InuyashaYang/AIDIY?style=social)](https://github.com/InuyashaYang/AIDIY)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1.0, lora_dropout=0.0, merge_weights=True):
        """
        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            r (int): 低秩矩阵的秩。
            lora_alpha (float): 缩放因子。
            lora_dropout (float): Dropout 概率。
            merge_weights (bool): 是否在微调后合并权重。
        """
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights

        # 冻结原始线性层的权重
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        # LoRA 参数
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_features))
            self.lora_B = nn.Parameter(torch.randn(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Scaling
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

    def forward(self, x):
        result = self.linear(x)
        if self.r > 0:
            delta = self.dropout(x) @ self.lora_A.t()  # (batch, r)
            delta = delta @ self.lora_B.t()            # (batch, out_features)
            result += delta * self.scaling
        return result

    def merge(self):
        """
        将 LoRA 参数合并到原始权重中。
        """
        if self.r > 0 and self.merge_weights:
            delta_W = self.lora_B @ self.lora_A
            self.linear.weight.data += delta_W * self.scaling
            # Optionally, remove LoRA parameters to save space
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

import math

# 示例用法
if __name__ == "__main__":
    in_features = 128
    out_features = 256
    batch_size = 32
    r = 8  # 低秩矩阵秩

    # 随机输入
    x = torch.randn(batch_size, in_features)

    # 实例化 LoRA 适配的线性层
    lora_linear = LoRALinear(in_features, out_features, r=r, lora_alpha=1.0, lora_dropout=0.1)

    # 前向传播
    output = lora_linear(x)
    print(f'输出形状: {output.shape}')  # 应为 (batch_size, out_features)

    # 查看可训练参数
    print("可训练的参数:")
    for name, param in lora_linear.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.shape}')
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
