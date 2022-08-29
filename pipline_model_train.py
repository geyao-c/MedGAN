import os
from util import toolsf

# 该代码文件的作用为依次选用不同的图片集训练模型
if __name__ == '__main__':
    root = "/home/lenovo/code/AdaGan/result/gnrted_img/2022-08-29-00:01:24/"
    AdaGAN_root = os.path.join(root, 'AdaGAN')
    WGAN_root = os.path.join(root, 'WGAN')
    s, d, e = 2500, 2500, 50000

    while s <= e:
        AdaGAN_gnrt_path = os.path.join(AdaGAN_root, str(s) + '-4')
        WGAN_gnrt_path = os.path.join(WGAN_root, str(s) + '-4')
        valid_path = '/home/lenovo/dataset/medical/processed_dataset/20-valid-4'

        # 使用AdaGAN生成的图片进行训练
        AdaGAN_gnrt_train = "python model_train.py --train_directory {} --valid_directory {} " \
                            "--gnrt_type {} --iter {} --optimizer_type SGD --lr 0.1 " \
                            "--epochs 120 --num_classes 4".format(AdaGAN_gnrt_path, valid_path, 'AdaGAN', s)
        cmd_list = [AdaGAN_gnrt_train] * 3
        toolsf.execute_command(cmd_list)

        # 使用WGAN生成的图片进行训练
        WGAN_gnrt_train = "python model_train.py --train_directory {} --valid_directory {} " \
                            "--gnrt_type {} --iter {} --optimizer_type SGD --lr 0.1 " \
                            "--epochs 120 --num_classes 4".format(WGAN_gnrt_path, valid_path, 'WGAN', s)
        cmd_list = [WGAN_gnrt_train] * 3
        toolsf.execute_command(cmd_list)

        s += d
