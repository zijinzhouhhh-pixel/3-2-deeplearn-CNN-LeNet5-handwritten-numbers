import torch
from torch import nn

# 定义一个网络模型
class MyLeNet5(nn.Module):
    # 初始化网络
    def __init__(self):
        super(MyLeNet5, self).__init__()
        # 定义网络层

        # 对于c1层，输入通道数：1；输出通道数：6；卷积核大小：5
        # 这里的 padding 是2
        self.c1 = nn.Conv2d(1, 6, 5, padding=2)
        self.relu = nn.ReLU()

        # 定义池化层 S2，这里采用的是平均池化，池化层宽度是2，步长是2，可以把图像宽度减半
        self.s2 = nn.AvgPool2d(2, 2)

        # 对于c3层，输入通道是6也就是s2的通道数，输出通道16，卷积核大小是5
        self.c3 = nn.Conv2d(6, 16, 5)

        # 定义池化层 s4，也是平均池化，池化层宽度是2，步长是2，也是可以把图像宽度减半
        self.s4 = nn.AvgPool2d(2, 2)

        # 定义卷积层 c5，输入通道是16，输出通道是120
        self.c5 = nn.Conv2d(16, 120, 5)

        # 需要把2维信息展开，展开为1维的，然后连接全连接层
        self.flatten = nn.Flatten()

        # 设置全连接层 f6
        # 输入特征图：120，输出特征图：84
        self.f6 = nn.Linear(120, 84)

        # 输出层:10
        self.output = nn.Linear(84, 10)

    # 向前传播
    # 卷积层和全连接层都需要激活函数
    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = self.relu(self.c5(x))
        x = self.flatten(x)
        x = self.relu(self.f6(x))
        x = self.output(x)
        return x

if __name__ == "__main__":
    # 默认输入都是一个随机张量的形式：批次是1，通道是1，大小是28*28
    # 4个值含义：批次/通道数/高度/宽度
    # x.shape[0]=1(Batch_Size，批次图片数量)
    # x.shape[1]=1(Channels，图像通道数)
    # x.shape[2]=28(图片高度)
    # x.shape[3]=28(图片宽度)
    x = torch.randn(1, 1, 28, 28)
    # 网络实例化
    model = MyLeNet5()
    # 输出y：手写数字的一个值
    y = model(x)