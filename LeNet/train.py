from model import Model, quanModel, Lenet5
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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.4734), (0.2507)),
    ])
    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4734), (0.2507)),
    ])
    torch.manual_seed(0)
    g = torch.Generator()
    train_loader = DataLoader(CIFAR10(root='../ResNet/cifar-10-python/', train=True, transform=transform_train),
                                          shuffle=True, generator=g, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(CIFAR10(root='../ResNet/cifar-10-python/', train=False, transform=transform_test),
                                 batch_size=batch_size, num_workers=0)


    # model = Lenet5().to(device)
    model = quanModel().to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    prev_acc = 0
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            opt.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            opt.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y =torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc), flush=True)
        if not os.path.isdir("models"):
            os.mkdir("models")
        
        model_quan = torch.quantization.convert(model)
        torch.save(model_quan, 'models/mnist_quan{:.3f}.pth'.format(acc))
        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc
    print("Model finished training")
