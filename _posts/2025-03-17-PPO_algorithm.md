---
title: PPO算法
description: >-
  RL基础
author: cybotiger
date: 2025-03-17 12:00:00 +0800
categories: [RL, Algorithm]
tags: [强化学习]
pin: true
math: true
mermaid: true
---

## 1. 传统策略梯度算法

### **1.1 从价值近似到策略近似**

强化学习算法可以分为两大类：基于值函数的强化学习和基于策略的强化学习。

- **基于值函数的强化学习**通过递归地求解贝尔曼方程来维护Q值函数（可以是离散的列表，也可以是神经网络），每次选择动作时会选择该状态下对应Q值最大的动作，使得未来积累的期望奖励值最大。这些算法在学习后的Q值函数不再发生变化，每次做出的策略也是一定的，可以理解为确定性策略。policy 如下：
    
    $$
    \pi:s \rightarrow a
    $$
    
- **基于策略的强化学习**不再通过价值函数来确定选择动作的策略，而是直接学习策略本身，通过一组参数 $\theta$ 对策略进行参数化，并通过神经网络方法优化。policy 如下：
    
    $$
    \pi_\theta(a|s)=P(a|s;\theta)
    $$
    

### **1.2 定义目标函数**

$$
\max_{\theta} J(\theta) = \max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}} R(\tau) = \max_{\theta} \sum_{\tau} P(\tau;\theta)R(\tau)
$$

其中轨迹 $\tau$ 是agent与环境交互产生的状态-动作轨迹 $(s_1,a_1,s_2,a_2,...)$，服从 $\pi_\theta$ 的概率分布