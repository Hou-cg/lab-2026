# lab-2026

2026 实验室招新前期考核 · 任务一：开发环境配置

作者：Hou-cg
日期：2026-10-08

---

## 一、我的电脑环境

| 项目 | 信息 |
|------|------|
| 操作系统 | Windows 11 家庭中文版 25H2（64 位） |
| CPU | Intel Core Ultra 5 235HX（十四核） |
| 显卡 | NVIDIA GeForce RTX 5060 Laptop GPU（8GB）+ Intel 集显 |
| 内存 | 16GB DDR5 5600MHz |

## 二、软件版本一览

| 软件 | 版本 | 安装位置 |
|------|------|----------|
| Miniconda | conda 4.9.2 | D:\Miniconda3 |
| Python | 3.11 | 虚拟环境 lab 中 |
| PyTorch | 2.14.1+cu130 | 虚拟环境 lab 中 |
| torchvision | 0.29.1+cu130 | 虚拟环境 lab 中 |
| torchaudio | 2.11.0+cu130 | 虚拟环境 lab 中 |
| Git | 2.56.0.windows.2 | C:\Program Files\Git（安装时使用默认路径） |
| 显卡驱动支持的 CUDA 版本 | 13.0 | 随系统 |

## 三、实际执行的安装步骤

### 1. 安装 Miniconda

- 下载来源：清华镜像 https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/
- 下载文件：Miniconda3-latest-Windows-x86_64.exe
- 安装位置：D:\Miniconda3（选择 D 盘，是为了不占用 C 盘系统盘空间，且后续 PyTorch 和数据集体积较大）
- 安装时勾选：Add Miniconda3 to my PATH environment variable
- 验证：`conda --version` → 输出 conda 4.9.2

### 2. 创建并激活虚拟环境

```bash
conda create -n lab python=3.11 -y
conda activate lab
```

命令提示符前出现 `(lab)`，表示已进入该环境。

说明：虚拟环境相当于一个独立的房间，项目所需工具都装在里面，与其他项目隔离，出问题可以直接删掉重建，不会影响整个系统。

### 3. 安装 PyTorch

先查看显卡支持情况：`nvidia-smi` → 右上角显示 CUDA Version: 13.0

安装命令：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

说明：必须加 `--index-url` 参数。因为 pip 默认从 PyPI 下载的是 CPU 版本，无法使用显卡；指定该地址才能装到 CUDA 版本。

### 4. 验证 GPU 是否可用

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出：

```
2.14.1+cu130
True
```

`True` 表示 PyTorch 已成功调用 NVIDIA 显卡，GPU 可用。

### 5. 安装 Git

- 下载来源：https://git-scm.com/download/win
- 安装位置：C:\Program Files\Git（安装时使用默认路径，未修改）
- 安装时默认编辑器选择 Notepad（避免误用 Vim 导致无法退出）
- 验证：`git --version` → git version 2.56.0.windows.2

### 6. Git 与 GitHub

- 注册 GitHub 账号：Hou-cg
- 创建公开仓库 lab-2026，初始化时包含 README.md，.gitignore 选择 Python 模板
- 克隆到本地：

```bash
cd /d/
git clone https://github.com/Hou-cg/lab-2026.git
```

说明：仓库选择 Public（公开），是为了让评审能够访问查看。

## 四、笔记必答（一）：如何判断该安装哪个版本的 PyTorch？

我的判断过程分为四步：

1. **先确认显卡型号**。通过系统信息工具查到我的显卡是 NVIDIA GeForce RTX 5060 Laptop GPU。

2. **再确认显卡驱动支持的 CUDA 版本**。执行 `nvidia-smi`，在输出表格右上角看到 `CUDA Version: 13.0`。这说明我的驱动最高支持到 CUDA 13.0。

3. **确认架构兼容性**。RTX 5060 属于较新的 Blackwell 架构（计算能力 sm_120），网上很多旧教程推荐的 cu118、cu121 版本并不支持这张显卡，装了会报 "not compatible with the current PyTorch installation" 的错误。因此需要选择 CUDA 12.8 以上、与驱动匹配的版本，最终选择 cu130。

4. **安装后必须验证**。用 `torch.cuda.is_available()` 检查，返回 `True` 才说明真的装对了。

结论：**先看显卡 → 再看驱动支持的 CUDA 版本 → 去 PyTorch 官网选对应的安装命令 → 最后用代码验证**。

## 五、笔记必答（二）：遇到的报错及解决方法

| 序号 | 报错 / 现象 | 原因 | 解决方法 |
|------|------------|------|----------|
| 1 | 教程里的下载链接打不开 | 域名写错了（实际不存在该地址） | 改用清华镜像站下载 |
| 2 | `'conda--version' 不是内部或外部命令` | 命令中间少了空格 | 改为 `conda --version` |
| 3 | `No command 'conda creat'` | 单词拼写错误（creat → create） | 按提示更正为 `conda create` |
| 4 | `The target prefix is the base prefix` | 漏写了选项前的短横线（`-n`） | 改为 `conda create -n lab python=3.11 -y` |
| 5 | `WARNING: Connection interrupted while downloading` | 下载中途网络波动 | 无需处理，pip 会自动断点续传，最终安装成功 |
| 6 | github.com 打不开 | 该网站在国内访问不稳定 | 安装 Watt Toolkit（开源工具）对 GitHub 进行网络加速 |
| 7 | `URL rejected: Bad hostname` | 命令里的 `.` 打成了 `,` | 更正地址为 `github.com` |

## 六、AI 使用说明

本次任务中，我使用 AI 工具辅助理解术语、生成命令和排查报错。

**AI 帮助到我的地方**：

- 解释 Miniconda、虚拟环境、PATH、Git、仓库等概念；
- 提示 RTX 5060 属于新架构，不能用旧的 cu118/cu121 版本；
- 提供 `nvidia-smi` 和 `torch.cuda.is_available()` 的验证思路。

**AI 出错 / 没解决的地方（如实记录）**：

1. AI 一开始给出的 Miniconda 下载链接是无效域名，直接打不开。后来我自己在搜索引擎中找到清华镜像站才完成下载。这说明 AI 给出的链接不能盲信，需要自己验证。
2. AI 第一次给出的环境创建命令写的是 python=3.10，与后续说明不完全一致，我在实际操作中改为 3.11，并按自己的环境调整。

**我的做法**：AI 给的每一条命令，我都会先看懂它是什么意思，再实际执行并用命令输出验证结果；遇到报错会把原文反馈，而不是反复乱试。

## 七、学习记录与个人感受

- 10/07：安装 Miniconda（D:\Miniconda3）、创建 lab 虚拟环境、安装 PyTorch、确认 GPU 可用（True）、安装 Git。踩过的坑：命令少空格、拼写错误、下载中断、GitHub 打不开。
- 10/08：注册 GitHub 账号，创建 lab-2026 仓库，克隆到本地，编写本 README，提交并推送。

个人感受：第一次接触终端，很多概念和操作都不是很熟练，也出了很多错，但基本在 AI 协助下完成大体部分，感觉很有挑战性也很新颖。
