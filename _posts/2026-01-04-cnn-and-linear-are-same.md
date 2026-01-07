---
title: 卷积网络和线性层的等价性
description: 卷积网络和线性层的等价性
author: cybotiger
date: 2026-01-04 08:00:00 +0800
categories: [AI, 神经网络]
tags: []
---

对于一个非重叠（non-overlapping）的卷积网络，参数为

```python
nn.Conv2d(
    input_channel=inc,
    output_channel=outc,
    kernel_size=p,
    stride=p
)
```

从 CNN 角度来看，输入为 `[inc, h, w]` 的图片张量，输出为 `[outc, h/p, w/p]` 的张量（p 整除 h,w）；按照定义，有 `outc` 组卷积核组，每组有 `inc` 个 `p*p` 大小的卷积核，分别和每个 patch 的对应 channel 卷积后求和，得到 `outc` 个输出 channel，对应 1 个 patch

而如果从线性层的角度来看，可以看作输入为 `[h/p*w/p, inc*p*p]` ，输出为 `[h/p*w/p, outc]`，即将每一个 patch 视为向量，共有 `h/p*w/p` 个向量，向量的输入维度为 `inc*p*p`，输出维度为 `outc*p*p`；于是线性层矩阵的形状为 `[inc*p*p, outc]`

这个矩阵共有 `outc` 列，每列的参数 为 对应组的 `inc` 个卷积核按序展平为一个 `inc*p*p` 的向量

## example
输入 `[3, 4, 4]` 的图片，输出 `[2, 2, 2]` 的张量，卷积网络和线性矩阵为
```python
# inc=3, outc=2, p=2
[
    [
     [1,1,
      1,1]
      ,                 # indim=12, outdim=2
     [1,1,              [
      1,1]                  [1, -1],
      ,                     [1, -1],
     [1,1,                  [1, -1],
      1,1]                  [1, -1],
    ]                       [1, -1],
    ,                       [1, -1],
                ->          [1, -1],
    [                       [1, -1],
     [-1,-1,                [1, -1],
      -1,-1]                [1, -1],
      ,                     [1, -1],
     [-1,-1,                [1, -1]
      -1,-1]            ]
      ,
     [-1,-1,
      -1,-1]
    ]
]
```

## 实际场景
由此可以得出一个结论，使用 CNN 对图像进行卷积 等价于 patchify+使用 linear 层进行变换；

在 Kaiming He 的 [JiT](https://arxiv.org/pdf/2511.13720) 工作中，论文中使用 linear transformation 变换图片，而在实际代码中使用的是 Conv2d

![alt text](assets/img/misc/jit-model.png)

```python
class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x
```