---
title: Github deploy key 的使用
description: 通过 deploy key 在远程服务器安全的管理单个仓库
author: cybotiger
date: 2025-12-01 07:00:00 +0800
categories: [网络, 工具]
tags: []
---

## 动机
在实验室的服务器上 push 项目代码，可以选择 https 协议或 ssh 协议。由于 https 协议在 push 时要求输入莫名其妙的 password，我尝试了账户密码、 patoken(personal access token)，都被服务器拒绝了。看了一下 stackoverflow 和 csdn，也是没有标准的解法。索性就全部使用 ssh 连接了。

## 步骤
而 ssh 连接，需要考虑安全性。我只是实习生，以后离开实验室之后要保护自己的其他无关仓库不被 hack，因此不能直接配置**关联账户**的 ssh key，而应该配置**关联单个仓库**的 deploy key。步骤如下：

首先，打开仓库的 settings，找到 security 下的 deploy key，添加新的 deploy key

然后在 server 端生成新的密钥（或使用已有的密钥）

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

之后 terminal 会询问你密钥对的存储位置，如果你不想覆盖已有的密钥对，需要自定义文件位置。

我们假定文件位于 `~/.ssh/id_rsa`，此时通过 `cat ~/.ssh/id_rsa.pub`，打印公钥，复制粘贴到 github 的 deploy key 中，就添加了新的 deploy key

之后尝试在 server 端连接 

```bash
ssh -T git@github.com
```

初次建立连接，会有询问信息，输入 yes 后，应该会看到以下内容，则配置成功，可以安全的 push 代码！

```
Hi USERNAME/REPO! You've successfully authenticated, but GitHub does not provide shell access.
```

最后，将 github 仓库的 remote 改成 ssh 连接

```bash
git remote set-url origin git@github.com:USERNAME/REPO.git
```
## 多个仓库
多个仓库无法共享同一个 deploy key！需要为每一个仓库生成新的密钥对。

同时，为了确保 ssh 连接 github 时，不同的仓库连接使用不同的密钥对，需要在 `~/.ssh/config` 文件中为不同的仓库域名配置别名，例如下：

```bash
Host github.com-sglang
        Hostname github.com
        IdentityFile=/home/your-username/.ssh/id_rsa

Host github.com-vllm
        Hostname github.com
        IdentityFile=/home/your-username/.ssh/id_rsa_1
``` 

然后可以测试仓库和对应 deploy key 的连接

```bash
ssh -T -i ~/.ssh/id_rsa_1 git@github.com-vllm
```

因为 ssh 连接默认使用 id_rsa 密钥，所以每个仓库的 remote url 应该配置为域名别名

```bash
git remote set-url origin git@github.com-vllm:USERNAME/VLLM-REPO.git

git remote set-url origin git@github.com-sglang:USERNAME/SGLANG-REPO.git
```

