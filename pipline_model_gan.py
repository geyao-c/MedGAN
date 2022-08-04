import datetime
import time
import subprocess
import argparse
import os
from util import toolsf

def argsget():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="other", help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default="/home/lenovo/dataset/medical/processed_dataset/gen", help='path to dataset')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--T1', type=float, default=0.3)
    parser.add_argument('--T2', type=float, default=0.5)
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = argsget()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    root = os.path.join('./result', now)
    AdaGAN_root = os.path.join(root, 'AdaGan')
    WGAN_root = os.path.join(root, 'WGAN')

    class_name_list = ["blister", "hydatoncus", "Demodicosis", "parakeratosis", "papillomatosis", "molluscum"]
    cmd_list = []
    for class_name in class_name_list:
        WGAN_cmd = "python WGAN_new.py --dataset {} --dataroot {} --batchSize {} --img_size {} --class_name {} " \
                   "--experiment {}".format(opt.dataset, opt.dataroot, opt.batchSize, opt.img_size, class_name, WGAN_root)
        AdaGAN_cmd = "python AdaGAN_new.py --dataset {} --dataroot {} --batchSize {} --img_size {} --class_name {} " \
                     "--T1 {} --T2 {} --experiment {}".format(opt.dataset, opt.dataroot, opt.batchSize, opt.img_size,
                                                              class_name, opt.T1, opt.T2, AdaGAN_root)
        cmd_list.append(WGAN_cmd)
        cmd_list.append(AdaGAN_cmd)
        if len(cmd_list) >= 12:
            toolsf.execute_command(cmd_list)
            cmd = []