### 一、训练AdaGan_new

**训练blister**

```python
# 本地
python AdaGAN_New.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 64 --img_size 64 --class_name blister --T1 0.25 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name blister --T1 0.3 --T2 0.5
```

**训练Demodicosis**

```python
# 本地
python AdaGAN_New.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 64 --img_size 64 --class_name Demodicosis --T1 0.25 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name Demodicosis --T1 0.3 --T2 0.5
```

**训练parakeratosis**

```python
# 本地
python AdaGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name parakeratosis --T1 0.3 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name parakeratosis --T1 0.3 --T2 0.5
```

**训练hydatoncus**

```python
# 本地
python AdaGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name hydatoncus --T1 0.3 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name hydatoncus --T1 0.3 --T2 0.5
```

**训练molluscum**

```python
# 本地
python AdaGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name molluscum --T1 0.3 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name molluscum --T1 0.3 --T2 0.5
```

**训练papillomatosis**

```python
# 本地
python AdaGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name papillomatosis --T1 0.3 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name papillomatosis --T1 0.3 --T2 0.5
```



### 二、训练WGan_new

**训练blister**

```python
# 本地
python WGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name blister 

# ubuntu服务器
python WGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name blister 
```

**训练Demodicosis**

```python
# 本地
python WGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name Demodicosis 

# ubuntu服务器
python WGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name Demodicosis 
```

**训练parakeratosis**

```python
# 本地
python WGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name parakeratosis 

# ubuntu服务器
python WGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name parakeratosis 
```

**训练hydatoncus**

```python
# 本地
python WGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name hydatoncus 

# ubuntu服务器
python WGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name hydatoncus 
```

**训练molluscum**

```python
# 本地
python WGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name molluscum 

# ubuntu服务器
python WGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name molluscum 
```

**训练papillomatosis**

```python
# 本地
python WGAN_new.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name papillomatosis 

# ubuntu服务器
python WGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name papillomatosis 
```



### 三、训练resnet50

```python
# 本地
python resnet_50_train.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name papillomatosis 

# ubuntu服务器
python model_train.py --train_directory /home/lenovo/dataset/medical/processed_dataset/gen --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid --logpath ./result/resnet_50_train/test/logger.log --optimizer_type SGD --lr 0.1 --epochs 120

python resnet_50_train.py --train_directory /home/lenovo/dataset/medical/processed_dataset/gen --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid --logpath ./result/resnet_50_train/test/logger.log --optimizer_type Adam --lr 0.0001

python model.py --train_directory /home/lenovo/code/AdaGan/result/gnrted_img/2022-08-04-17:58:18/AdaGAN/50000 --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid --logpath ./result/resnet_50_train/test/logger.log --optimizer_type Adam --lr 0.0001
    
python model_train.py --train_directory /home/lenovo/code/AdaGan/result/gnrted_img/2022-08-04-17:58:18/AdaGAN/50000 --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid --logpath ./result/resnet_50_train/test/logger.log --optimizer_type SGD --lr 0.1
    
python model_train.py --train_directory /home/lenovo/code/AdaGan/result/gnrted_img/2022-08-04-17:58:18/WGAN/50000 --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid --logpath ./result/resnet_50_train/test/logger.log --optimizer_type SGD --lr 0.1
```



### 四、训练resnet18

```python
# ubuntu服务器
python model_train.py --train_directory /home/lenovo/dataset/medical/processed_dataset/gen --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid --logpath ./result/resnet_50_train/test/logger.log --optimizer_type SGD --lr 0.1 --epochs 120

# 使用生成的图片进行训练 AdaGan
python model_train.py --train_directory /home/lenovo/code/AdaGan/result/gnrted_img/2022-08-04-17:58:18/AdaGAN/50000-4 --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid4 --logpath ./result/resnet_18_train/logger.log --optimizer_type SGD --lr 0.1
    
python model_train.py --train_directory /home/lenovo/code/AdaGan/result/gnrted_img/2022-08-04-17:58:18/AdaGAN/50000-2 --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid2 --logpath ./result/resnet_18_train/logger.log --optimizer_type SGD --lr 0.1

# 使用生成的图片进行训练 WGAN
python model_train.py --train_directory /home/lenovo/code/AdaGan/result/gnrted_img/2022-08-04-17:58:18/WGAN/50000-4 --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid4 --logpath ./result/resnet_18_train/logger.log --optimizer_type SGD --lr 0.1

# 使用原始图片进行训练
python model_train.py --train_directory /home/lenovo/dataset/medical/processed_dataset/gen4 --valid_directory /home/lenovo/dataset/medical/processed_dataset/valid4 --logpath ./result/resnet_18_train/logger.log --optimizer_type SGD --lr 0.1
```

