---
title: Diffusion model
description: >-
  diffusion原理与架构
author: cybotiger
date: 2025-05-21 12:00:00 +0800
categories: [AI]
tags: []
math: true
mermaid: true
---

## 简介
diffusion 的思想是通过对一个纯噪声去噪来生成样本（预期的图片）。其过程分为 forward process 和 reverse process，前者将真实数据集中的图像逐步添加噪声，最终生成纯噪声；后者通过对 forward 过程添加噪声的逆过程（即去噪）进行建模，来生成样本。

forward 过程是假想的，实际并不会发生，因此加噪过程的 $q$ 分布是一个理论分布，需要我们建模来进行拟合，也就是去噪过程的 $p_\theta$

## 拟合概率分布 $q$
加噪过程的概率分布 $q$ 如下计算：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1})
$$

我们的拟合概率分布 $p_\theta$ 希望从 $x_t$ 去噪生成 $x_{t-1}$，然而无法直接进行去噪获得 $q(\mathbf{x}_{t-1}\|\mathbf{x}_t)$ ；此处通过加入 $x_0$ 的 condition 使得逆向过程可解（可建模），即：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}\tilde{\beta}_t\mathbf{I})
$$

经化简，得：

$$
\tilde{\beta}_t = \color{green}\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t
$$

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_t\right)
$$