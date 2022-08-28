import torch
import os
import torchvision.utils as vutils
import time
import datetime
import subprocess
import logging

def sg_fk_img_gnrt(model, noise, dir, begin_idx=0):
    # 创建文件夹
    if not os.path.exists(dir):
        os.makedirs(dir)

    with torch.no_grad():
        fake = model(noise)
    fake.data = fake.data.mul(0.5).add(0.5)
    image_num = fake.shape[0]

    for i in range(image_num):
        image_path = os.path.join(dir, '{}.png'.format(begin_idx + i + 1))
        sgimage = fake[i, :, :, :]
        print(sgimage.shape)
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
    bc_fk_img_gnrt(model, noise, bc_fk_dir, iter)

def model_save(netG, netD, iter, dir):
    model_save_dir = os.path.join(dir, 'save_model')

    # 创建文件夹
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    G_save_path = os.path.join(model_save_dir, 'netG_iter{}.pth'.format(iter))
    D_save_path = os.path.join(model_save_dir, 'netD_iter{}.pth'.format(iter))
    torch.save(netG.state_dict(), G_save_path)
    torch.save(netD.state_dict(), D_save_path)

def isok(sub_list):
    for item in sub_list:
        if item.poll() is None:
            return False
    return True

def execute_command(cmdstring_list, cwd=None, timeout=None, shell=True):
    """执行一个SHELL命令
        封装了subprocess的Popen方法, 支持超时判断，支持读取stdout和stderr
        参数:
      cwd: 运行命令时更改路径，如果被设定，子进程会直接先更改当前路径到cwd
      timeout: 超时时间，秒，支持小数，精度0.1秒
      shell: 是否通过shell运行
    Returns: return_code
    Raises: Exception: 执行超时
    """
    # if shell:
    #     cmdstring_list = cmdstring
    # else:
    #     cmdstring_list = shlex.split(cmdstring)
    if timeout:
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

    sub_list = []
    # 没有指定标准输出和错误输出的管道，因此会打印到屏幕上；
    for i, item in enumerate(cmdstring_list):
        sub = subprocess.Popen(item, cwd=cwd, stdin=subprocess.PIPE, shell=shell, bufsize=4096)
        time.sleep(0.5)
        sub_list.append(sub)

    # subprocess.poll()方法：检查子进程是否结束了，如果结束了，设定并返回码，放在subprocess.returncode变量中
    print('开始执行')
    while True:
        if isok(sub_list) is True: break
        time.sleep(0.5)
    print('执行完了')
    # return str(sub.returncode)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_logger(file_path):

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

# 构建logger和writer
def lgwt_construct(logpath):
    # 建立日志
    logger = get_logger(logpath)  # 运行时日志文件
    return logger