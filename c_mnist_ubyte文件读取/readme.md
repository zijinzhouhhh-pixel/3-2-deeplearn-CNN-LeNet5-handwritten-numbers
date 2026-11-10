# mnist 数据文件可视化

把 HTML 文件拖入到网页中去，然后上传对应的文件内容，即可获得可视化效果

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

## 操作步骤

注意：这里是 .gz 即可

1. 把html文件拖入浏览器
2. 分别选取train数据中的：train-images-idx3-ubyte.gz，train-labels-idx1-ubyte.gz
3. 可视化展示
