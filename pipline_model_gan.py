import datetime
import time
import subprocess
import argparse
import os

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
        time.sleep(5)
        sub_list.append(sub)

    # subprocess.poll()方法：检查子进程是否结束了，如果结束了，设定并返回码，放在subprocess.returncode变量中
    print('开始执行')
    while True:
        if isok(sub_list) is True: break
        time.sleep(0.5)
    print('执行完了')
    # return str(sub.returncode)

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
            execute_command(cmd_list)
            cmd = []