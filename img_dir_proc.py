import os
import shutil

categories = ['blister', 'Demodicosis', 'parakeratosis', 'molluscum']

def sol(dirpath):
    s, d, e = 2500, 2500, 50000
    while s <= e:
        for category in categories:
            source_dir = os.path.join(dirpath, str(s), category)
            target_dir = os.path.join(dirpath, str(s) + '-4', category)
            # 首先创建新的文件夹
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            # 从源文件夹中拷贝图片到目标文件夹中
            filelist = os.listdir(source_dir)
            for file in filelist:
                filename = os.path.join(source_dir, file)
                # 将文件从源路劲拷贝到目标路劲中
                shutil.copy(filename, target_dir)
        print('{}拷贝完毕'.format(s))
        s += d

# 这份代码的作用是构建只包含四种医疗图片的文件夹
if __name__ == '__main__':
    root = "/home/lenovo/code/AdaGan/result/gnrted_img/2022-08-29-00:01:24/"
    # 得到具体的图片目录
    AdaGAN_root = os.path.join(root, 'AdaGAN')
    WGAN_root = os.path.join(root, 'WGAN')
    sol(AdaGAN_root)
    sol(WGAN_root)


