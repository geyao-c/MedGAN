import torch
import os
import torchvision.utils as vutils

def sg_fk_img_gnrt(model, noise, dir):
    # 创建文件夹
    if not os.path.exists(dir):
        os.makedirs(dir)

    with torch.no_grad():
        fake = model(noise)
    fake.data = fake.data.mul(0.5).add(0.5)
    image_num = fake.shape[0]

    for i in range(image_num):
        image_path = os.path.join(dir, '{}.png'.format(i + 1))
        sgimage = fake[i, :, :, :]
        vutils.save_image(sgimage, image_path)

def bc_rl_img_gnrt(real_data, dir, iter):
    # 创建文件夹
    if not os.path.exists(dir):
        os.makedirs(dir)

    real_cpu = real_data.mul(0.5).add(0.5)
    image_path = os.path.join(dir, '{}_real_samples.png'.format(iter))
    vutils.save_image(real_cpu, image_path)

def bc_fk_img_gnrt(model, noise, dir, iter):
    # 创建文件夹
    if not os.path.exists(dir):
        os.makedirs(dir)

    with torch.no_grad():
        fake = model(noise)
    fake.data = fake.data.mul(0.5).add(0.5)
    fake_image_path = os.path.join(dir, '{}_fake_samples.png'.format(iter))
    vutils.save_image(fake.data, fake_image_path)

def timage_gnrt(model, real_data, noise, root, iter):
    # 在单个文件夹下生成多张图片
    sg_fk_dir = os.path.join(root, 'sg_fk_img', str(iter))
    sg_fk_img_gnrt(model, noise, sg_fk_dir)

    # 生成一批次真图片
    bc_rl_dir = os.path.join(root, 'bc_img')
    bc_rl_img_gnrt(real_data, bc_rl_dir, iter)

    # 生成一批次假图片
    bc_fk_dir = os.path.join(root, 'bc_img')
    bc_fk_img_gnrt(model, noise, iter)
