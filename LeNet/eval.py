import torch
import torch.quantization
import numpy as np 
from model import Model  # 导入你的模型类
import os 
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import Module
from torch import nn


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
model = Model()
model = torch.load("models/mnist_0.982.pkl")
print_size_of_model(model)
model.eval()  # 量化前，需要确保模型处于 eval 模式
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


print_size_of_model(quantized_model)
# torch.jit.save(quantized_model, 'models/mnist_quan.pt')
# -- 量化前后的精度对比 --
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256
train_dataset = mnist.MNIST(root='./data', train=True, transform=ToTensor())
test_dataset = mnist.MNIST(root='./data', train=False, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
all_correct_num = 0
all_sample_num = 0
for idx, (test_x, test_label) in enumerate(test_loader):
    test_x = test_x.to(device)
    test_label = test_label.to(device)
    predict_y = model(test_x.float()).detach()
    predict_y =torch.argmax(predict_y, dim=-1)
    current_correct_num = predict_y == test_label
    all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
    all_sample_num += current_correct_num.shape[0]
print('accuracy: {:.3f}'.format(all_correct_num / all_sample_num), flush=True)



all_correct_num = 0
all_sample_num = 0
# print(1)
quantized_model = torch.load("models/mnist_quan0.972.pth")
# model = quanModel()
# device = 'cpu'
quantized_model.eval()
quantized_model.to(device)
# print(1)
with torch.no_grad():
    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_label = test_label.to(device)
        predict_y = quantized_model(test_x.float()).detach()
        predict_y =torch.argmax(predict_y, dim=-1)
        current_correct_num = predict_y == test_label
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]

print('accuracy: {:.3f}'.format(all_correct_num / all_sample_num), flush=True)