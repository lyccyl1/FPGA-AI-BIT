import numpy as np
from torchvision.models import resnet152
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def evaluate(model, data_loader):
    model.eval()
    pred_tags = []
    true_tags = []
    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch[0].to(device)
            batch_label = batch[1].to(device)

            logits = model(batch_data)

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            tags = batch_label.cpu().numpy()

            pred_tags.extend(pred)
            true_tags.extend(tags)

    assert len(pred_tags) == len(true_tags)
    correct_num = sum(int(x == y) for (x, y) in zip(pred_tags, true_tags))
    accuracy = correct_num / len(pred_tags)

    return accuracy

def train(model, data_loader, loss_func):
    model.train()
    for batch in tqdm(data_loader):
        data = batch[0].to(device)
        label = batch[1].to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, label)
        
        loss.backward()
        optimizer.step()

model = resnet152()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 10)
model.to(device)

transform_model = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.449), (0.226)),
    ])

transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.4734), (0.2507))
    ])

torch.manual_seed(0)
g = torch.Generator()
train_dataloader = DataLoader(CIFAR10(root='./cifar-10-python/', train=True, transform=transform_model, download=True),
                                        shuffle=True, generator=g, batch_size=32, num_workers=4)
test_dataloader = DataLoader(CIFAR10(root='./cifar-10-python/', train=False, transform=transform_test),
                            batch_size=32, num_workers=4)
optimizer = Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

max_test_acc = 0.
for epoch in range(100):
    
    train(model, train_dataloader, loss_func)

    torch.save(model.state_dict(), './models/resnet152_last.pt')
    test_acc = evaluate(model, test_dataloader)
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        torch.save(model.state_dict(), './models/resnet152_best.pt')
        print("Best model saved!")

    print("epoch: {}  test_acc: {:.2f}%".format(epoch, test_acc * 100))

