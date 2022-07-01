import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from time import time
import os
import json

import models.dcgan as dcgan
import models.mlp as mlp
import numpy as np
import copy as cp
import datetime
import util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
one = torch.FloatTensor([1]).to(device)
mone = (one * - 1).to(device)

def argsget():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="other", help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default="samples", help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=1100000, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    # parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default='./result/WGAN/', help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--classid', type=str, default=None)
    opt = parser.parse_args()
    return opt

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def generate(netG, fixed_noise_gen, epoch, device, root):
#     # global fixed_noise_gen
#     netG.to(device), fixed_noise_gen.to(device)
#     fake = netG(fixed_noise_gen)
#     fake.data = fake.data.mul(0.5).add(0.5)
#     image_generate_dir = os.path.join(root, 'generate_image')
#     for i in range(generate_num):
#         img1 = fake.data[i, ...].reshape((1, opt.nc, imageSize, imageSize))
#         img2 = cp.deepcopy(img1)
#         img2[0, 1, :, :] = img1[0, 0, :, :]
#         img2[0, 2, :, :] = img1[0, 0, :, :]
#         # vutils.save_image(img2, os.path.join(root_path + "generate/" + str(epoch) + "/", str(i) + ".jpg"))
#         vutils.save_image(img2, os.path.join(image_generate_dir, str(epoch), str(i) + ".jpg"))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def discriminator_train(netD, data, noise, optimizerD, opt):
    # 设置参数可更新
    for p in netD.parameters():
        p.requires_grad = True

    # clamp参数
    for p in netD.parameters():
        p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

    netD.zero_grad()

    # train with real
    real = data.to(device)
    errD_real = netD(real)
    errD_real.backward(one)

    # train with fake
    noise.resize_(opt.batchSize, opt.nz, 1, 1).normal_(0, 1)
    # generator生成fake
    fake = netG(noise).detach().clone()
    errD_fake = netD(fake)
    errD_fake.backward(mone)
    errD = errD_real - errD_fake
    # 更新参数
    optimizerD.step()

    return errD_real, errD_fake, errD

def generator_train(netG, noise, optimizerG, opt):
    # 清空梯度
    netG.zero_grad()
    # 生成fake
    noise.resize_(opt.batchSize, opt.nz, 1, 1).normal_(0, 1)
    fake = netG(noise)
    # 得到输出
    errG = netD(fake)
    errG.backward(one)
    optimizerG.step()

    return errG

if __name__ == '__main__':
    opt = argsget()
    # 当前时间
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    root = os.path.join(opt.experiment, now)
    # 图片存储文件夹
    image_generate_dir = os.path.join(root, 'image_generate')
    mkdir(image_generate_dir)
    record_dir = os.path.join(root, 'record')
    mkdir(record_dir)
    # 日志文件
    rf = open(os.path.join(record_dir, 'log.txt'), "w")

    # 设置随机种子
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    # 加载数据集
    dataloader = util.dstget(opt)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": opt.nz, "nc": opt.nc, "ngf": opt.ngf, "ngpu": opt.ngpu,
                        "n_extra_layers": opt.n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(record_dir, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config) + "\n")

    # 构建generator
    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

    # generator参数初始化
    netG.apply(weights_init)
    if opt.netG != '':  # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # write out generator config to generate images together wth training checkpoints (.pth)
    discriminator_config = {"imageSize": opt.imageSize, "nz": opt.nz, "nc": opt.nc, "ngf": opt.ndf, "ngpu": opt.ngpu,
                        "n_extra_layers": opt.n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(record_dir, "discriminator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(discriminator_config) + "\n")

    # 构建discriminator
    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, opt.nz, opt.nc, opt.ndf, opt.ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, opt.nz, opt.nc, opt.ndf, opt.ngpu, opt.n_extra_layers)
        netD.apply(weights_init)

    # discriminator参数初始化化
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print("netD ", netD)

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize).to(device)
    # tensor默认requires_grad == False
    noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).to(device)
    fixed_noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1).normal_(0, 1).to(device)
    netD.to(device)
    netG.to(device)

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    gen_iterations = 0
    # discriminator, generate训练轮数
    total_D, total_G, total_DG = 0, 0, 0
    errD_real, errD_fake, errD, errG = None, None, None, None

    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            # Diter: discriminator训练轮次
            if gen_iterations < 25 or gen_iterations % 500 == 0: Diters = 100
            else: Diters = opt.Diters
            j = 0
            # 更新discriminator
            while j < Diters and i < len(dataloader):
                j += 1; total_D += 1; total_DG = total_D + total_G

                data = data_iter.next()[0].to(device)
                i += 1

                # 训练discriminator
                errD_real, errD_fake, errD = discriminator_train(netD, data, noise, optimizerD, opt)

            for p in netD.parameters():
                p.requires_grad = False

            # 更新generator
            errG = generator_train(netG, noise, optimizerG, opt)

            total_G += 1
            total_DG = total_D + total_G

            # 写日志
            errD_real, errD_fake, errD, errG = round(errD_real.cpu().item(), 2), round(errD_fake.cpu().item(), 2), \
                                               round(errD.cpu().item(), 2), round(errG.cpu().item(), 2)
            rf.write("total_DG: {}, gen_iterations: {}, errD_real: {}, errD_fake: {}, errD: {}, errG: {}\n".
                     format(total_DG, gen_iterations, errD_real, errD_fake, errD, errG))
            print("total_DG: {}, gen_iterations: {}, errD_real: {}, errD_fake: {}, errD: {}, errG: {}".
                  format(total_DG, gen_iterations, errD_real, errD_fake, errD, errG))
            gen_iterations += 1

            # 生成图片
            if total_DG % 5000 == 0:
                real_cpu = data[0].mul(0.5).add(0.5)
                real_image_path = os.path.join(image_generate_dir, '{}_real_samples.png'.format(total_DG))
                vutils.save_image(real_cpu, real_image_path)

                fake = netG(fixed_noise)
                fake.data = fake.data.mul(0.5).add(0.5)
                fake_image_path = os.path.join(image_generate_dir, '{}_fake_samples.png'.format(total_DG))
                vutils.save_image(fake.data, fake_image_path)
