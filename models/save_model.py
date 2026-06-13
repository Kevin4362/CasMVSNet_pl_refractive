import sys

sys.path.append("/home/kevin/CasMVSNet_pl")
import torch
from models.mvsnet import CascadeMVSNet

# 1. 创建模型
model = CascadeMVSNet()

# 2. 保存模型（只保存结构+权重，最简单版本）
torch.save(model, "model.pth")

print("✅ 保存成功！文件：model.pth")
print("👉 直接拖到 Netron 里就能看完整网络！")
