import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import os
import torchvision.transforms as tr

class myDataset(data.Dataset):
    def __init__(self, opt):
        # 图片root dir
        self.root = opt.dataroot
        # 具体某一种类型的图片文件夹
        self.class_name = opt.class_name
        self.img_size = opt.img_size

        self.sroot_dir = os.path.join(self.root, self.class_name)
        self.file_paths = [os.path.join(self.sroot_dir, file) for file in os.listdir(self.sroot_dir)]

        self.normal = tr.Compose([
            tr.ToPILImage(),
            tr.ToTensor(),
            tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        item = self.file_paths[index]
        # opencv读取图片
        Img = cv2.imread(item)
        # 改变图片尺寸
        Img = cv2.resize(Img, (self.img_size, self.img_size))
        # 从BGR通道转换为RGB通道
        Img = Img[:, :, (2, 1, 0)]
        Img = self.normal(Img)
        Img = np.asarray(Img)
        Img = Img.astype(np.float)
        Input_tensor = torch.from_numpy(Img.astype(np.float)).type(torch.FloatTensor)
        return Input_tensor
