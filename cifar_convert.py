import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.4734), (0.2507)),
])
torch.manual_seed(0)
g = torch.Generator()

test_loader = DataLoader(CIFAR10(root='../ResNet/cifar-10-python/', train=False, transform=transform_test),
                                batch_size=1, num_workers=0)
device = 'cpu'
f1 = open("cifar-10-python/data.txt", "w", encoding="utf-8")
f2 = open("cifar-10-python/label.txt", "w", encoding="utf-8")
for idx, (train_x, train_label) in enumerate(test_loader):
            train_x = list(train_x.to(device)[0][0].flatten().numpy())
            train_label = list(train_label.to(device).numpy())
            line_data = ",".join(map(str, train_x))+",\n"
            line_label = ",".join(map(str, train_label))+","
            f1.write(line_data)
            f2.write(line_label)
            
