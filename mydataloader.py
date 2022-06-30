import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import os
import torchvision.transforms as tr

img_size=64
print("WGAN生成",img_size)
class myDataset(data.Dataset):
    def __init__(self,class_id):
        # self.normal = tr.Compose([
        #                    tr.ToPILImage(),
        #                    tr.ToTensor(),
        #                    tr.Normalize(mean = (0.4685483813116096, 0.538136651819416, 0.6217816988531444), std = (0.1016119525359456, 0.0900060860845122, 0.08024531900661314))
        #                ])
        self.normal=tr.Compose([
                                # tr.Scale(img_size),
                                tr.ToPILImage(),
                                tr.ToTensor(),
                                tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        '''
        
        root="/root/userfolder/mydestiny/AdaGAN/txt/"
        with open(root+'gan_train_'+str(class_id)+'.txt') as f:
            self.file_paths=f.read().splitlines()
            print('train number:',len(self.file_paths))
        '''
        '''
        毛囊虫生成
        root = "../../train/1/"
        self.file_paths = [root+f for f in os.listdir(root)]
        print('train number:', len(self.file_paths))
        '''
        root = "G:/data/TemperatureDomain/divdedbycat/train/J113/"
        # root = "./data/cifar10"
        self.file_paths = [root + f for f in os.listdir(root)]
        print(self.file_paths)
        print('train number:', len(self.file_paths))


    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, index):
        item = self.file_paths[index]
        Img = cv2.imread(item)
        Img = cv2.resize(Img, (img_size, img_size))
        Img = Img[:, :, (2, 1, 0)]
        Img = self.normal(Img)
        Img = np.asarray(Img)
        Img = Img.astype(np.float)
        Input_tensor = torch.from_numpy(Img.astype(np.float)).type(torch.FloatTensor)
        return Input_tensor
