
import json
import torch
from pathlib import Path
from model import MyLeNet5


# ==================================================
# 1. 获取项目根目录
# ==================================================
# 当前脚本：
# C:\Study_zjz\1-AI\machinelearn\deeplearn\2-LeNet-5-YannLeCun1998\d_export_weights.py

BASE_DIR = Path(__file__).resolve().parent

print("=" * 50)
print("LeNet-5 模型权重导出")
print("=" * 50)

print("项目目录：", BASE_DIR)


# ==================================================
# 2. 定义模型权重路径
# ==================================================

weight_path = (
    BASE_DIR
    / "save_LeNet5_model"
    / "model-LeNet5.pth"
)

if not weight_path.exists():
    raise FileNotFoundError(
        f"模型权重文件不存在：\n{weight_path}"
    )

print("权重文件：", weight_path)


# ==================================================
# 3. 加载 LeNet-5 模型
# ==================================================

model = MyLeNet5()

state_dict = torch.load(
    weight_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)

# 切换为推理模式
model.eval()

print("\n模型加载成功！")


# ==================================================
# 4. 导出模型全部参数
# ==================================================

weights = {}

for name, tensor in model.state_dict().items():

    # 将 Tensor 转换为 Python 列表
    weights[name] = tensor.detach().cpu().tolist()

print("\n权重导出完成！")
print("权重数量：", len(weights))

print("\n模型参数名称：")

for name, tensor in weights.items():
    print(name)


# ==================================================
# 5. 导出为 JavaScript 文件
# ==================================================

output_path = BASE_DIR / "weights.js"

with open(output_path, "w", encoding="utf-8") as f:

    # 定义全局 JavaScript 变量
    f.write("window.MODEL_WEIGHTS = ")

    # 写入 JSON 数据
    json.dump(
        weights,
        f,
        ensure_ascii=False,
        separators=(",", ":")
    )

    f.write(";\n")


# ==================================================
# 6. 输出导出结果
# ==================================================

print("\n" + "=" * 50)
print("模型权重导出成功！")
print("=" * 50)

print("模型：MyLeNet5")
print("权重文件：", weight_path.name)
print("输出文件：", output_path)
print("文件大小：", round(output_path.stat().st_size / 1024, 2), "KB")
print("=" * 50)
