# experiment_mnist.py
# 姓名：侯程罡    学号：202600192
# 程序简介：对比实验。用同一份代码、靠参数切换配置，跑三组对照：
#           ① 训练轮数 1/5/10   ② 隐藏层 64/128/256   ③ 优化器 Adam vs SGD
#           每组记录测试集准确率和训练耗时，最后汇总成表格，并写入 results.md。

import os  # 拼路径
import time  # 计时

import torch  # 张量、自动求导
import torch.nn as nn  # 网络层与损失函数
import torch.optim as optim  # 优化器
from torchvision import datasets, transforms  # 数据集与图像变换

# ===== 固定不变的设置 =====
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # 数据目录
BATCH_SIZE = 64  # 每批 64 张
SEED = 0  # 随机种子，保证每次初始化、打乱顺序都一样，结果可复现
INPUT_SIZE = 784  # 28*28，输入固定
OUTPUT_SIZE = 10  # 10 个类别，输出固定
ADAM_LR = 0.001  # Adam 的学习率
SGD_LR = 0.01  # SGD 的学习率（SGD 用 0.001 会学得太慢，故用常用默认值 0.01）
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.md")  # 结果文件


class Net(nn.Module):
    """可定制隐藏层大小的两层全连接网络：784 -> hidden -> 10。"""

    def __init__(self, hidden_size):
        """构造函数：按传入的隐藏层大小建层。

        参数 hidden_size：隐藏层神经元个数。
        返回：None。
        """
        super().__init__()  # 初始化父类
        self.hidden = nn.Linear(INPUT_SIZE, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.output = nn.Linear(hidden_size, OUTPUT_SIZE)  # 隐藏层到输出层

    def forward(self, x):
        """前向传播：输入 (batch, 784)，输出 (batch, 10) 的得分。

        参数 x：一批展平后的图片。
        返回：10 个类别的得分。
        """
        return self.output(self.relu(self.hidden(x)))  # 线性 -> ReLU -> 线性


def load_data():
    """加载 MNIST，返回训练和测试两个 DataLoader（只加载一次，供所有配置复用）。

    参数：无。
    返回：训练集 loader 和测试集 loader 的二元组。
    """
    to_tensor = transforms.ToTensor()  # 像素缩放成 0~1 的张量
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=to_tensor)  # 训练集
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)  # 测试集
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # 训练集打乱
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  # 测试集不打乱
    return train_loader, test_loader


def build_model(hidden_size):
    """按种子创建网络，保证每个配置的初始权重完全相同。

    参数 hidden_size：隐藏层神经元个数。
    返回：初始化好的网络。
    """
    torch.manual_seed(SEED)  # 固定种子，保证不同配置的网络初始权重一致
    return Net(hidden_size)


def train(model, train_loader, optimizer, epochs):
    """训练指定轮数，返回平均 loss 和纯训练耗时（秒）。

    参数 model：要训练的网络。
    参数 train_loader：训练数据。
    参数 optimizer：优化器。
    参数 epochs：训练轮数。
    返回：平均 loss 与训练耗时的二元组。
    """
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    model.train()  # 训练模式
    start = time.time()  # 开始计时（只计训练，不计数据加载和评估）
    total_loss = 0.0  # 累计损失
    step_count = 0  # 累计批次数

    for _ in range(epochs):  # 训练 epochs 轮
        for images, labels in train_loader:  # 逐批取数据
            images = images.view(images.size(0), INPUT_SIZE)  # 展平成 (batch, 784)
            outputs = model(images)  # 前向
            loss = criterion(outputs, labels)  # 算损失
            optimizer.zero_grad()  # 清空上一批的梯度
            loss.backward()  # 反向传播算梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累计损失
            step_count += 1  # 累计批次

    train_time = time.time() - start  # 训练耗时
    return total_loss / step_count, train_time


def evaluate(model, test_loader):
    """在测试集上算准确率。

    参数 model：训练好的网络。
    参数 test_loader：测试数据。
    返回：准确率（0~1）。
    """
    model.eval()  # 评估模式
    correct = 0  # 猜对的数量
    total = 0  # 总数量
    with torch.no_grad():  # 评估不算梯度
        for images, labels in test_loader:  # 逐批
            images = images.view(images.size(0), INPUT_SIZE)  # 展平
            predicted = model(images).argmax(dim=1)  # 取得分最大的类别
            correct += (predicted == labels).sum().item()  # 累加猜对数
            total += labels.size(0)  # 累加总数
    return correct / total  # 准确率


def run_one_config(hidden_size, epochs, optimizer_name, train_loader, test_loader):
    """跑一个配置：建网络 -> 训练 -> 测试，返回一行结果。

    参数 hidden_size：隐藏层大小。
    参数 epochs：训练轮数。
    参数 optimizer_name："adam" 或 "sgd"。
    参数 train_loader：训练数据。
    参数 test_loader：测试数据。
    返回：包含配置、准确率、耗时、参数量的字典。
    """
    model = build_model(hidden_size)  # 建网络（种子固定）
    if optimizer_name == "adam":  # 选优化器
        optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)
        lr_used = ADAM_LR
    else:
        optimizer = optim.SGD(model.parameters(), lr=SGD_LR)
        lr_used = SGD_LR

    average_loss, train_time = train(model, train_loader, optimizer, epochs)  # 训练
    accuracy = evaluate(model, test_loader)  # 测试
    params = sum(p.numel() for p in model.parameters())  # 统计参数量
    return {"hidden": hidden_size, "epochs": epochs, "optimizer": optimizer_name,
            "lr": lr_used, "accuracy": accuracy, "train_time": train_time,
            "loss": average_loss, "params": params}


def make_config_list():
    """构造三组实验配置（去重后返回待跑列表）。

    参数：无。
    返回：列表，每项是 (hidden, epochs, optimizer) 三元组。
    """
    configs = [
        (128, 1, "adam"),  # 第一组：轮数对比
        (128, 5, "adam"),
        (128, 10, "adam"),
        (64, 5, "adam"),  # 第二组：隐藏层大小对比
        (256, 5, "adam"),
        (128, 5, "sgd"),  # 第三组：优化器对比
    ]
    return configs  # (128,5,adam) 会被复用，不重复跑


def main():
    """主流程：跑所有配置 -> 打印分组表格 -> 写 results.md。

    参数：无。
    返回：None。
    """
    train_loader, test_loader = load_data()  # 数据只加载一次
    print("数据加载完成：训练 %d 批，测试 %d 批" % (len(train_loader), len(test_loader)))
    print()

    configs = make_config_list()  # 待跑配置
    cache = {}  # 结果缓存，键是 (hidden, epochs, optimizer)
    for hidden, epochs, optimizer_name in configs:  # 逐个跑
        print(">>> 开始: 隐藏层=%d, 轮数=%d, 优化器=%s" % (hidden, epochs, optimizer_name.upper()))
        result = run_one_config(hidden, epochs, optimizer_name, train_loader, test_loader)  # 跑一个配置
        cache[(hidden, epochs, optimizer_name)] = result  # 存起来
        print("    准确率 %.2f%%   训练耗时 %.2f 秒   参数量 %d" % (
            result["accuracy"] * 100, result["train_time"], result["params"]))
        print()

    # 打印分组表格
    print("=" * 78)
    print("第一组：训练轮数（隐藏层 128，Adam）")
    print("=" * 78)
    print("%-8s %-12s %-10s" % ("轮数", "准确率", "训练耗时(秒)"))
    for epochs in (1, 5, 10):
        r = cache[(128, epochs, "adam")]
        print("%-8d %-12s %-10.2f" % (epochs, "%.2f%%" % (r["accuracy"] * 100), r["train_time"]))

    print()
    print("=" * 78)
    print("第二组：隐藏层大小（训练 5 轮，Adam）")
    print("=" * 78)
    print("%-10s %-12s %-14s %-10s" % ("隐藏层", "准确率", "参数量", "训练耗时(秒)"))
    for hidden in (64, 128, 256):
        r = cache[(hidden, 5, "adam")]
        print("%-10d %-12s %-14d %-10.2f" % (hidden, "%.2f%%" % (r["accuracy"] * 100), r["params"], r["train_time"]))

    print()
    print("=" * 78)
    print("第三组：优化器（训练 5 轮，隐藏层 128）")
    print("=" * 78)
    print("%-10s %-10s %-12s %-10s" % ("优化器", "学习率", "准确率", "训练耗时(秒)"))
    for name in ("adam", "sgd"):
        r = cache[(128, 5, name)]
        print("%-10s %-10s %-12s %-10.2f" % (name.upper(), r["lr"], "%.2f%%" % (r["accuracy"] * 100), r["train_time"]))

    write_results(cache)  # 写 markdown
    print()
    print("结果已写入:" + RESULTS_PATH)


def write_results(cache):
    """把结果写成 markdown 文件。

    参数 cache：键为配置、值为结果字典的缓存。
    返回：None。
    """
    lines = []  # 逐行收集
    lines.append("# MNIST 对比实验结果\n")
    lines.append("由 `experiment_mnist.py` 自动生成。数据：MNIST 测试集 10000 张；batch_size = 64；随机种子 = 0。\n")
    lines.append("## 第一组：训练轮数（隐藏层 128，Adam，lr=0.001）\n")
    lines.append("| 训练轮数 | 测试集准确率 | 训练耗时(秒) |")
    lines.append("|---|---|---|")
    for epochs in (1, 5, 10):
        r = cache[(128, epochs, "adam")]
        lines.append("| %d | %.2f%% | %.2f |" % (epochs, r["accuracy"] * 100, r["train_time"]))

    lines.append("\n## 第二组：隐藏层大小（训练 5 轮，Adam，lr=0.001）\n")
    lines.append("| 隐藏层大小 | 参数量 | 测试集准确率 | 训练耗时(秒) |")
    lines.append("|---|---|---|---|")
    for hidden in (64, 128, 256):
        r = cache[(hidden, 5, "adam")]
        lines.append("| %d | %d | %.2f%% | %.2f |" % (hidden, r["params"], r["accuracy"] * 100, r["train_time"]))

    lines.append("\n## 第三组：优化器对比（训练 5 轮，隐藏层 128）\n")
    lines.append("| 优化器 | 学习率 | 测试集准确率 | 训练耗时(秒) |")
    lines.append("|---|---|---|---|")
    for name in ("adam", "sgd"):
        r = cache[(128, 5, name)]
        lines.append("| %s | %s | %.2f%% | %.2f |" % (name.upper(), r["lr"], r["accuracy"] * 100, r["train_time"]))

    lines.append("\n## 说明\n")
    lines.append("- 训练耗时只统计算训练循环的时间，不含数据加载与测试评估。")
    lines.append("- 每个配置都用相同随机种子（0）初始化，保证起点一致，差异只来自配置本身。")
    lines.append("- 两个优化器各用各自常用的学习率：Adam 0.001、SGD 0.01（SGD 用 0.001 收敛太慢，不适合直接对比）。")
    lines.append("- 隐藏层 128、5 轮、Adam 这个配置同时出现在第一、二、三组，只实际跑了一次，三张表共用同一份结果。")

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:  # 写文件
        fh.write("\n".join(lines) + "\n")  # 每行一份，末尾补换行


main()  # 程序入口
