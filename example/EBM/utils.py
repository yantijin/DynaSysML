import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def plot_fig(res, path=None, n=6, show=False):
    # n = 6
    digit_size = 28
    figure = np.zeros((28 * n, 28 * n))
    for i in range(n):
        for j in range(n):
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = res[n * i + j, 0, :, :]

    plt.figure()
    if show:
        plt.imshow(figure, cmap='gray')
    if path:
        plt.savefig(path)
    return figure

def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0, num=0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='../data/', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=num, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='../data/', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='../data/', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader

def write_to_file(data, path):
    with open(path, 'a+') as file_obj:
        file_obj.write(' '.join([str(i) for i in data]))
        file_obj.write('\n')

def read_file(path):
    ls = []
    with open(path, 'r') as file_obj:
        for line in file_obj.readlines():
            ls.append([float(i) for i in line.split(' ')])
    ls = torch.tensor(np.array(ls), dtype=torch.float32)
    return ls