import os.path

import torch
import argparse
import models.dcgan as dcgan
import models.mlp as mlp
from util import toolsf

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
"""
python imge_gnrt.py --netG_path ./model/netG_iter50000.pth --gnrt_num 1000 --img_saved_dir ./result/image
"""

def argsget():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--netG_path', type=str, default=None, help='generator path')
    parser.add_argument('--gnrt_num', type=int, default=1000, help='generator image number')
    parser.add_argument('--num_each_batch', type=int, default=32, help='num of each batch')
    parser.add_argument('--img_saved_dir', type=str, default=None, help='img saved dir')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = argsget()

    # 构建generator
    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.img_size, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.img_size, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.img_size, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.n_extra_layers)

    # 加载模型
    if opt.netG_path is None:
        raise('netG path is None')
    map_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # map_str = 'cpu'
    netG.load_state_dict(torch.load(opt.netG_path, map_location=torch.device(map_str)))
    netG.to(device)

    if opt.img_saved_dir is None:
        raise ('netG path is None')
    if not os.path.exists(opt.img_saved_dir):
        os.makedirs(opt.img_saved_dir)

    gnrted_num = 0
    while gnrted_num < opt.gnrt_num:
        batch_num = min(opt.num_each_batch, opt.gnrt_num - gnrted_num)
        noise = torch.FloatTensor(batch_num, opt.nz, 1, 1).normal_(0, 1).to(device)
        toolsf.sg_fk_img_gnrt(netG, noise, opt.img_saved_dir, gnrted_num)
        # toolsf.bc_fk_img_gnrt(netG, noise, opt.img_saved_dir, gnrted_num)
        gnrted_num += batch_num




