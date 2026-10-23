# train_mnist.py
# 姓名：侯程罡    学号：202600192
# 程序简介：手搭一个两层全连接神经网络（784 -> 128 -> 10）训练 MNIST，
#           训练 1 个 epoch，batch_size = 64，最后在测试集上算准确率。
#           本文件手写：网络结构、训练循环、测试评估，不使用任何现成模型。

import os  # 用于拼接数据目录路径
import time  # 用于统计训练耗时

import torch  # PyTorch 主库：张量、自动求导
import torch.nn as nn  # nn 模块：定义网络层和损失函数
import torch.optim as optim  # optim 模块：优化器（这里用 Adam）
from torchvision import datasets, transforms  # datasets 加载 MNIST，transforms 转张量

# ===== 常量（集中放在这里，避免代码里出现"魔术数字"）=====
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # 数据目录：脚本同目录下的 data
INPUT_SIZE = 784  # 输入层神经元数 = 28 * 28（把图片展平成 784 个数）
HIDDEN_SIZE = 128  # 隐藏层神经元数
OUTPUT_SIZE = 10  # 输出层神经元数 = 10 个类别（数字 0~9）
BATCH_SIZE = 64  # 每个 batch 装多少张图片
EPOCHS = 1  # 先只训练 1 轮
LEARNING_RATE = 0.001  # Adam 的学习率（最常用的默认值）


class SimpleNet(nn.Module):
    """手动定义的两层全连接神经网络：784 -> 128 (ReLU) -> 10。"""

    def __init__(self):
        """构造函数：在这里创建网络需要的层。

        参数：无。
        返回：None。
        """
        super().__init__()  # 先初始化父类 nn.Module，PyTorch 要求必须这么做
        self.hidden = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)  # 第 1 层：784 个输入 -> 128 个隐藏神经元
        self.relu = nn.ReLU()  # 激活函数：把负数变成 0，正数保持不变（引入非线性）
        self.output = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)  # 第 2 层：128 个隐藏神经元 -> 10 个输出

    def forward(self, x):
        """前向传播：给定一批输入，算出 10 个类别的得分。

        参数 x：形状为 (batch, 784) 的张量，每行是一张展平后的图片。
        返回：形状为 (batch, 10) 的张量，每个数字是该类别的得分（logits）。
        """
        x = self.hidden(x)  # 线性变换：x 乘以权重再加偏置，变成 128 维
        x = self.relu(x)  # 套一层 ReLU 激活
        x = self.output(x)  # 再线性变换一次，变成 10 维
        return x  # 返回 10 个类别的得分


def load_data():
    """加载 MNIST 训练集和测试集，并套上 DataLoader 方便按 batch 取数据。

    参数：无。
    返回：两个二元组 (train_loader, test_loader)。
    """
    to_tensor = transforms.ToTensor()  # 把 0~255 的整数像素缩放成 0.0~1.0 的浮点张量
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=to_tensor)  # 训练集
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_tensor)  # 测试集
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # 训练集按 64 张一批、每轮打乱顺序
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  # 测试集按 64 张一批、不打乱
    return train_loader, test_loader  # 交回两个数据加载器


def train_one_epoch(model, train_loader, criterion, optimizer):
    """训练 1 个 epoch：把训练集所有图片过一遍并更新参数。

    参数 model：要训练的网络。
    参数 train_loader：训练集的数据加载器。
    参数 criterion：损失函数（交叉熵）。
    参数 optimizer：优化器（Adam）。
    返回：这一轮的平均损失。
    """
    model.train()  # 切换到"训练模式"（本网络没有 Dropout/BatchNorm，但仍按规范加上）
    total_loss = 0.0  # 累加每个 batch 的损失，用来算平均
    batch_count = 0  # 记录处理了多少个 batch，用来算平均

    for images, labels in train_loader:  # 每次取出一批图片和对应标签
        images = images.view(images.size(0), INPUT_SIZE)  # 把 (64,1,28,28) 展平成 (64, 784)，作为网络输入
        outputs = model(images)  # 前向传播：得到这批图片的 10 个类别得分
        loss = criterion(outputs, labels)  # 用交叉熵算这批的损失

        optimizer.zero_grad()  # 清空上一步残留的梯度（不清会累加）
        loss.backward()  # 反向传播：自动算出每个参数的梯度
        optimizer.step()  # 优化器按梯度更新一次参数

        total_loss += loss.item()  # 累加损失值（.item() 把它变成普通 Python 数字）
        batch_count += 1  # batch 计数加一
        print("batch %4d   loss = %.4f" % (batch_count, loss.item()))  # 每个 batch 打印一次 loss

    return total_loss / batch_count  # 返回这一轮的平均损失


def evaluate(model, test_loader):
    """在测试集上算准确率。

    参数 model：训练好的网络。
    参数 test_loader：测试集的数据加载器。
    返回：准确率（0.0 ~ 1.0 之间的小数）。
    """
    model.eval()  # 切换到"评估模式"
    correct = 0  # 猜对的张数
    total = 0  # 一共多少张

    with torch.no_grad():  # 评估时不需要算梯度，关掉可以省内存、更快
        for images, labels in test_loader:  # 每次取一批图片和标签
            images = images.view(images.size(0), INPUT_SIZE)  # 同样展平成 (64, 784)
            outputs = model(images)  # 前向传播得到得分
            predicted = outputs.argmax(dim=1)  # 取得分最大的那个位置当预测结果
            correct += (predicted == labels).sum().item()  # 这一批里猜对的数量，累加
            total += labels.size(0)  # 这一批的图片数量，累加

    return correct / total  # 猜对的占比就是准确率


def main():
    """主程序：加载数据 -> 建网络 -> 训练 1 轮 -> 测试准确率。

    参数：无。
    返回：None。
    """
    torch.manual_seed(0)  # 固定随机种子，保证每次打乱顺序、初始化权重都一样（方便复现）

    train_loader, test_loader = load_data()  # 第 1 步：准备数据
    print("训练集 batch 数:", len(train_loader))  # 打印训练集一共有多少个 batch（60000/64 = 938）
    print("测试集 batch 数:", len(test_loader))  # 打印测试集一共有多少个 batch（10000/64 = 157）
    print()

    model = SimpleNet()  # 第 2 步：创建网络
    criterion = nn.CrossEntropyLoss()  # 损失函数：交叉熵（内部已含 softmax，不用自己加）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 优化器：Adam，更新网络的所有参数

    print("开始训练 %d 个 epoch ..." % EPOCHS)  # 提示开始
    start = time.time()  # 记录开始时间
    average_loss = train_one_epoch(model, train_loader, criterion, optimizer)  # 第 3 步：训练
    elapsed = time.time() - start  # 统计训练耗时
    print()
    print("训练完成：平均 loss = %.4f，耗时 %.2f 秒" % (average_loss, elapsed))  # 报告训练结果
    print()

    accuracy = evaluate(model, test_loader)  # 第 4 步：在测试集上评估
    print("测试集准确率: %.2f%%" % (accuracy * 100))  # 打印准确率（换成百分比更好读）


main()  # 程序入口：调用 main 函数开始运行
