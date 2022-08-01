### 一、训练AdaGan_new

**训练blister**

```python
# 本地
python AdaGAN_New.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 64 --img_size 64 --class_name blister --T1 0.25 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 64 --img_size 64 --class_name blister --T1 0.25 --T2 0.5
```

**训练Demodicosis**

```python
# 本地
python AdaGAN_New.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 64 --img_size 64 --class_name Demodicosis --T1 0.25 --T2 0.5

# ubuntu服务器
python AdaGAN_new.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name Demodicosis --T1 0.3 --T2 0.5
```

### 一、训练WGan_new

**训练blister**

```python
# 本地
python WGAN_New.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name blister 

# ubuntu服务器
python WGAN_New.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name blister 
```

**训练Demodicosis**

```python
# 本地
python WGAN_New.py --dataset others --dataroot /Users/chenjie/dataset/医疗数据集/processed_dataset/gen --batchSize 16 --img_size 64 --class_name Demodicosis 

# ubuntu服务器
python WGAN_New.py --dataset others --dataroot /home/lenovo/dataset/medical/processed_dataset/gen --batchSize 16 --img_size 64 --class_name Demodicosis 
```

