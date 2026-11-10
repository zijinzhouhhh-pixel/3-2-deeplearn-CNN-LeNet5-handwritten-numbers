"""
【从0开始撸代码--手把手教你搭建LeNet-5网络模型】https://www.bilibili.com/video/BV1vU4y1A7QJ?p=4&vd_source=9e6ceee45c2d3f68c855f593d8e6ee6b

"""

import torch
from torch import nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader

from model import MyLeNet5
from torch.optim import lr_scheduler
from a_download_data import train_dataset, test_dataset
import os
from pathlib import Path

# ==========================
# 1. 训练参数配置
# ==========================
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.01

# ==========================
# 2. 设备配置：如果有显卡，可以转化为GPU
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# 3. 数据加载器
# ==========================
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# 4. 模型初始化：调用net里面定义的模型，然后奖模型数据转到GPU上
# ==========================
model = MyLeNet5().to(device)

# ==========================
# 5. 损失函数（交叉熵损失函数）
# ==========================
loss_fn = nn.CrossEntropyLoss()

# ==========================
# 6. 优化器：SGD（Stochastic Gradient Descent ）随机梯度下降：深度学习中最经典的优化算法之一
# ==========================
# 参数 momentum，约定俗成0.9，它的核心作用是：让梯度下降具有一定的惯性，减少优化过程中的震荡，加快向最优解收敛的速度。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ==========================
# 7. 学习率调度器：每隔10轮变为原来的 0.1，这里是防止后续的抖动比较大，我们的学习率越小，可能越能找到损失最小的点
# ==========================
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==========================
# 8. 权重模型存储位置
# ==========================
# 获取当前 train.py 文件所在目录
BASE_DIR = Path(__file__).resolve().parent
# 模型保存目录
SAVE_DIR = BASE_DIR / "save_LeNet5_model"


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 定义：loss损失，精确度，n
    loss, current, n = 0.0, 0.0, 0
    # 取出来数据
    for batchm, (X, y) in enumerate(dataloader):
        # 【A-前向传播】
        # 向前传播进行训练：首先给到显卡 GPU 训练
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        # 损失函数：拿输出值和真实的标签，进行交叉熵的计算，根据损失值进行反向传播
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        # 这里是计算这一批次的精确度，如果y和预测的相同，就是1；如果不相同就是0。分母是这一批次的数据数量（）
        cur_acc = torch.sum(pred == y) / output.shape[0]

        # 【B-反向传播】
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()

        # 检查梯度
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.abs().mean().item())

        # 更新参数
        optimizer.step()

        # 把所有的损失加在一起
        # 把所有的精度加在一起
        loss += cur_loss.item()
        current += cur_acc.item()

        n += 1

    # 打印该批次的训练损失和训练精度
    print("train_loss：" + str(loss / n))
    print("train_acc：" + str(current / n))

def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batchm, (X, y) in enumerate(dataloader):
            # 【A-前向传播】
            # 向前传播进行训练：首先给到显卡 GPU 训练
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            # 损失函数：拿输出值和真实的标签，进行交叉熵的计算，根据损失值进行反向传播
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            # 这里是计算这一批次的精确度，如果y和预测的相同，就是1；如果不相同就是0。分母是这一批次的数据数量（）
            cur_acc = torch.sum(pred == y) / output.shape[0]
            # 把所有的损失加在一起
            # 把所有的精度加在一起
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1
        # 打印该批次的训练损失和训练精度
        print("val_loss：" + str(loss / n))
        print("val_acc：" + str(current / n))
        return current / n

# 开始训练
epoch = 50
min_acc = 0
for t in range(epoch):
    print(f" ")
    print(f"================== epoch{t+1} ==================")
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    scheduler.step()
    # 保存最好的模型权重
    if a > min_acc:
        # 如果文件夹不存在，则自动创建
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        min_acc = a
        print("save best model")
        # 保存权重模型
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model-LeNet5.pth"))

print('Done!')