"""
MNIST 数据集下载与加载模块
运行此脚本会自动下载 MNIST 数据集到 ./data 目录
"""
import torch
from torchvision import datasets, transforms
from pathlib import Path

# 获取当前 Python 文件所在目录
BASE_DIR = Path(__file__).resolve().parent
# 数据集存储目录
DATA_DIR = BASE_DIR / 'data'

# 数据转化为 tensor 格式：数据集肯定是数组格式，这里进入神经网络需要转化为张量
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 这里是数据集下载下来：加载 MNIST 手写数字训练集，并对数据进行预处理
# 加载训练数据集
train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, transform=data_transform, download=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, transform=data_transform, download=True)

if __name__ == "__main__":
    print(f"训练集数量: {len(train_dataset)}")
    print(f"测试集数量: {len(test_dataset)}")
    print("MNIST 数据集下载完成！")
