# 读取模型 并把权重参数保存下来
import torch
import numpy as np

the_model = torch.load("./models/mnist_0.983.pth")


for key, value in the_model.items(): # 由于最邻近差值没有参数，只需要将其他参数赋予给新模型即可
    the_model[key] = value
    f = open('./parameters/'+key+".txt", "w")

    value_cpu = value.cpu()
    data_arr=value_cpu.numpy()
    data_arr = data_arr.flatten()
    np.savetxt('./parameters/'+key+'.txt', data_arr, fmt='%f', delimiter='\r\n')
    # f.write(data_arr)