import os
import shutil

CH2EN = {
    "水疱": "blister",
    "囊肿": "hydatoncus",
	"毛囊虫": "Demodicosis",
    "角化不全": "parakeratosis",
    "乳头瘤样增生": "papillomatosis",
    "软疣小体": "molluscum"
}

CH2GNUM = {
    "水疱": 200,
    "囊肿": 320,
	"毛囊虫": 200,
    "角化不全": 500,
    "乳头瘤样增生": 500,
    "软疣小体": 160
}

# 每一类选用20张进行图片生成
GenNum = 20

def mrmdir(diroot):
    filelist = os.listdir(diroot)
    for file in filelist:
        filepath = os.path.join(diroot, file)
        # 删除文件夹
        if os.path.isdir(filepath):
            mrmdir(os.path.join(filepath))
        else:
            os.remove(filepath)
    os.rmdir(diroot)

if __name__ == '__main__':
    ori_data_root = "/Users/chenjie/dataset/医疗数据集/original_dataset"
    pro_data_root = "/Users/chenjie/dataset/医疗数据集/processed_dataset"
    dirlist = os.listdir(ori_data_root)
    print(dirlist)
    print(CH2EN.keys())

    for dir in dirlist:
        if dir in CH2EN.keys():
            # 分配图片数据
            dirpath = os.path.join(ori_data_root, dir)
            print(dirpath)
            filelist = os.listdir(dirpath)
            # print(len(filelist))
            pure_filelist = []
            for file in filelist:
                if ".jpg" in file or ".png" in file or ".bmp" in file:
                    pure_filelist.append(file)
            # print(len(pure_filelist))

            gen_tar_dir = os.path.join(pro_data_root, 'gen', CH2EN[dir])
            valid_tar_dir = os.path.join(pro_data_root, 'valid', CH2EN[dir])

            if not os.path.exists(gen_tar_dir):
                os.makedirs(gen_tar_dir)
            else:
                mrmdir(gen_tar_dir)
                os.makedirs(gen_tar_dir)

            if not os.path.exists(valid_tar_dir):
                os.makedirs(valid_tar_dir)
            else:
                mrmdir(valid_tar_dir)
                os.makedirs(valid_tar_dir)

            # 前Gennum张用于生成图片
            for i in range(0, GenNum):
                filename = pure_filelist[i]
                src_filepath = os.path.join(dirpath, filename)
                shutil.copy(src_filepath, gen_tar_dir)
            print(gen_tar_dir)

            # 剩下的图片用作验证
            for i in range(GenNum, CH2GNUM[dir]):
                filename = pure_filelist[i]
                src_filepath = os.path.join(dirpath, filename)
                shutil.copy(src_filepath, valid_tar_dir)
            print(valid_tar_dir)
