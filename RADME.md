# CNN神经网络学习: LeNet-5深度神经网络

本项目为：Yann Le Cun 在1998年实现的CNN网络架构
本质上：手写数字识别

## A-核心功能说明

具体教程案例，参照飞书文本：

```
CNN卷积神经网络：https://qcnwdgfqgzgz.feishu.cn/docx/IKBYdl5Roo4WBOxb74Mczk5Knre
```

```
LeNet-5学习：https://qcnwdgfqgzgz.feishu.cn/docx/HddcdRHScoRVG8x0Yz6cmv4Innf?previous_navigation_time=1794212644053&disposable_login_token=eyJ1c2VyX2lkIjoiNzQ2OTYwNTc5MTMzMTg2MDUwOCIsImRldmljZV9sb2dpbl9pZCI6Ijc1Mzk3MzEzNjQxMzI3MDAxNjMiLCJ0aW1lc3RhbXAiOjE3OTQyMTI2NDQsInVuaXQiOiJldV9uYyIsInB3ZF9sZXNzX2xvZ2luX2F1dGgiOiIxIiwidmVyc2lvbiI6InYzIiwidGVuYW50X2JyYW5kIjoiZmVpc2h1IiwicGtnX2JyYW5kIjoi6aOe5LmmIiwiY2xpZW50X3NjaGVtYSI6IngtZmVpc2h1In0%3D.a9a95708e584f71ad21f40135eb2203976bac7cebb68a4ac4d706720faca7910
```

```
视频教程：【从0开始撸代码--手把手教你搭建LeNet-5网络模型】https://www.bilibili.com/video/BV1vU4y1A7QJ?p=3&vd_source=9e6ceee45c2d3f68c855f593d8e6ee6b
```

### 1.神经网络的设计和实现

```
https://www.bilibili.com/video/BV1vU4y1A7QJ?p=3&vd_source=9e6ceee45c2d3f68c855f593d8e6ee6b
```

```
https://qcnwdgfqgzgz.feishu.cn/docx/HddcdRHScoRVG8x0Yz6cmv4Innf
```

### 2.训练数据的准备和处理

| 文件名                  | 含义       | 数据内容          |
| ----------------------- | ---------- | ----------------- |
| train-images-idx3-ubyte | 训练集图片 | 60,000 张手写数字 |
| train-labels-idx1-ubyte | 训练集标签 | 60,000 个数字标签 |
| t10k-images-idx3-ubyte  | 测试集图片 | 10,000 张手写数字 |
| t10k-labels-idx1-ubyte  | 测试集标签 | 10,000 个数字标签 |

ubyte 图片数据是怎么存储的？
每个像素使用一个 uint8 数值表示：

```
.ubyte 是一种二进制数据文件（Binary Data File），不是图片格式，也不是模型文件。
```

- 0：黑色。
- 255：白色。
- 1～254：不同程度的灰色。
  一张图片包含：28 个像素

### 3.模型的训练和测试流程

| 层      | 类型     | 需要ReLU？ | 原因                          |
| ------- | -------- | ---------- | ----------------------------- |
| c1      | 卷积层   | √          | 引入非线性，原版采用Sigmoid   |
| s2      | 池化层   | x          | 池化只是降采样，不需要激活    |
| c3      | 卷积层   | √          | 同c1                          |
| s4      | 池化层   | x          | 同s2                          |
| c5      | 卷积层   | √          | 同c1                          |
| flatten | 展平     | x          | 只是维度变换                  |
| f6      | 全连接层 | √          | 引入非线性                    |
| output  | 输出层   | x          | 输出 raw logits, 不能追加ReLU |

## B-基础资料来源

## C-文件说明output
