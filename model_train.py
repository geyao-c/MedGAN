import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from util import toolsf
import models
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def argsget():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_directory', type=str, default="/Users/chenjie/dataset/医疗数据集/processed_dataset/gen")
    parser.add_argument('--valid_directory', type=str, default="/Users/chenjie/dataset/医疗数据集/processed_dataset/valid")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--logpath', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--optimizer_type', type=str, default='SGD')


    opt = parser.parse_args()
    return opt

def train(model, loss_function, optimizer, train_data):
    model.train()

    losses = toolsf.AverageMeter('Loss', ':.4e')
    top1 = toolsf.AverageMeter('Acc@1', ':6.2f')

    for i, (inputs, labels) in enumerate(train_data):
        inputs = inputs.to(device)
        labels = labels.to(device)
        n = inputs.size(0)

        # 因为这里梯度是累加的，所以每次记得清零
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        prec1, prec5 = toolsf.accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg

def valid(model, loss_function, valid_data):
    losses = toolsf.AverageMeter('Loss', ':.4e')
    top1 = toolsf.AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        model.eval()
        for j, (inputs, labels) in enumerate(valid_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            n = inputs.size(0)
            # print(labels)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            print(loss)

            prec1, prec5 = toolsf.accuracy(outputs, labels, topk=(1, 5))

            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)

    return losses.avg, top1.avg

if __name__ == '__main__':
    opt = argsget()
    if opt.logpath is None:
        raise ('logpath is None')
    logger = toolsf.get_logger(opt.logpath)
    logger.info('args is %s', opt)

    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=80, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=80),
            transforms.CenterCrop(size=64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    }

    num_classes = 6
    data = {
        'train': datasets.ImageFolder(root=opt.train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=opt.valid_directory, transform=image_transforms['valid'])
    }
    logger.info('train data len: {}, valid data len: {}'.format(len(data['train']), len(data['valid'])))
    # 读取数据
    train_data = DataLoader(data['train'], batch_size=opt.batch_size, shuffle=True, num_workers=8)
    valid_data = DataLoader(data['valid'], batch_size=opt.batch_size, shuffle=True, num_workers=8)

    # 构造模型
    model = models.resnet.resnet18(pretrained=False, num_classes=num_classes)
    model.to(device)

    # 构造损失函数和优化器
    loss_func = nn.CrossEntropyLoss().to(device)

    # 定义优化器
    if opt.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)  # Adam梯度下降

    logger.info(model)
    # print(resnet50)

    start_epoch = 0
    best_top1_acc = 0

    # 训练模型
    epoch = start_epoch
    best_accu_model, valid_top1_acc = None, None

    lr = opt.lr
    while epoch < opt.epochs:
        print('--------------------------------------------------------------------------------')
        if opt.optimizer_type == 'SGD':
            # 学习率在0.5和0.75的时候乘以0.1
            if epoch in [int(opt.epochs * 0.5), int(opt.epochs * 0.75)]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    lr = param_group['lr']
        logger.info('lr is: {}'.format(lr))
        start = time.time()
        train_obj, train_top1_acc = train(model, loss_func, optimizer, train_data)
        valid_obj, valid_top1_acc = valid(model, loss_func, valid_data)
        # logstore(writer, train_obj, train_top1_acc, valid_obj, valid_top1_acc, epoch)

        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc

        epoch += 1
        end = time.time()
        logger.info("train loss is: {:.3f}, train accuracy is {:.3f}".format(train_obj, train_top1_acc))
        logger.info("valid loss is: {:.3f}, valid accuracy is {:.3f}".format(valid_obj, valid_top1_acc))
        logger.info("=>epoch:{}/{}, Best accuracy {:.3f} cost time is {:.3f}".format(epoch, opt.epochs, best_top1_acc, (end - start)))



