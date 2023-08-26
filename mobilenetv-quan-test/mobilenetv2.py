import torch
from torch.quantization import prepare_qat, get_default_qat_qconfig, convert
from torchvision.models import quantization

# Step1：修改模型
# 这里直接使用官方修改好的MobileNet V2，下文会对修改点进行介绍
model = quantization.mobilenet_v2()
print("original model:")
print(model)

# Step2：折叠算子
# fuse_model()在training或evaluate模式下算子折叠结果不同，
# 对于QAT，需确保在training状态下进行算子折叠
assert model.training
model.fuse_model()
print("fused model:")
print(model)

# Step3:指定量化方案
# 通过给模型实例增加一个名为"qconfig"的成员变量实现量化方案的指定
# backend目前支持fbgemm和qnnpack
BACKEND = "fbgemm"
model.qconfig = get_default_qat_qconfig(BACKEND)

# Step4：插入伪量化模块
prepare_qat(model, inplace=True)
print("model with observers:")
print(model)

# 正常的模型训练，无需修改代码

# Step5：实施量化
model.eval()
# 执行convert函数前，需确保模型在evaluate模式
model_int8 = convert(model)
print("quantized model:")
print(model_int8)

# Step6：int8模型推理
# 指定与qconfig相同的backend，在推理时使用正确的算子
torch.backends.quantized.engine = BACKEND
# 目前Pytorch的int8算子只支持CPU推理,需确保输入和模型都在CPU侧
# 输入输出仍为浮点数
fp32_input = torch.randn(1, 3, 224, 224)
y = model_int8(fp32_input)
print("output:")
print(y)