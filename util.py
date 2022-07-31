import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch
from gendataloader import myDataset


def dstget(opt):
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Scale(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root="./data/cifar10/", download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
                               )
    # 医学图片使用自定义dataloader加载
    else:
        dataset = myDataset(opt)
        # Dataloader_train = data.DataLoader(dataset, args.batch_size,
        #                             num_workers = args.num_workers,
        #                             shuffle = True, pin_memory = True)

    assert dataset
    # 加载数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    return dataloader