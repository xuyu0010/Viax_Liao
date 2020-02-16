# General Action Recognition Framework for Viax

This repository is the framework for Action Recognition for Viax Project.

## Implementation

This framework is in PyTorch

### Normalization
The inputs are substrated by mean RGB = [ 124, 117, 104 ], and then multiplied by 0.0167.


## Usage

Fine-tune with pre-trained model using the new network:
```
python train_ucf101.py --network CHANGE_1 (or CHANGE_2)
```
or 
```
python train_hmdb51.py --network CHANGE_1 (or CHANGE_2)
```

Evaluate the trained model:
```
cd test
# the default setting is to test trained model on ucf-101 (split1)
python evaluate_video.py --network CHANGE_1
```


## Other Resources

ImageNet-1k Trainig/Validation List:
- Download link: [GoogleDrive](https://goo.gl/Ne42bM)

ImageNet-1k category name mapping table:
- Download link: [GoogleDrive](https://goo.gl/YTAED5)

Kinetics Dataset:
- Downloader: [GitHub](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)

UCF-101 Dataset:
- Download link: [Website](http://crcv.ucf.edu/data/UCF101.php)

HMDB51 Dataset:
- Download link: [Website](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database)

## 额外中文说明

- 若没有显卡进行训练，请使用CPU训练。目前网络已经修改为采用CPU训练，若有Bug请记录并及时通过邮件和班主任反馈给我。若能使用GPU也请尽早反馈。
- 首先保证所有的对文件的链接改为文件夹，将视频文件（每个类的文件应在不同文件夹）都放入对应的数据库/raw/data文件夹内。
- 如果仍然遇到问题，可以尝试重装PyTorch。

### 如何修改网络框架

- 1. 保留原有骨架网络，新增一个网络py文件（比如：mfnet_change.py）注意文件名不要有空格，建议能体现修改的内容。
- 2. 建议修改网络的class名（见mfnet_base.py第65行）（改称比如 class MFNET_CHANGE(nn.Module):）。这个改动要同时体现在__init__函数的super内和main函数中。
- 3. 测试采用python 新文件名即可，main函数大致和mfnet_base.py的main函数一致，如果修改了网络名记得在main函数内做相应的修改。（如果碰到xxx is not defined很大可能就是你网络名没有修改）
- 4. 在symbol_builder.py中新加一个from 新文件 import 新网络class，并参考13，14行做相应的增加（可以直接先复制黏贴然后修改网络名字就好）
- 5. 在train_hmdb51.py中修改第38行 --network的参数为你修改过的新网络名

### 数据库下载链接如下（可在中国大陆使用）
- UCF101：https://entuedu-my.sharepoint.com/:u:/g/personal/xuyu0014_e_ntu_edu_sg/EaDYMqmoSPNHqM3sVyEH4jkBvPJP6fGDYBTX6n2vB0-jXA?e=I3OxfT
- HMDB51：https://entuedu-my.sharepoint.com/:u:/g/personal/xuyu0014_e_ntu_edu_sg/ETkMjaDh3DdCjaJ-PfX5tW8B043TZH26aeheP4laCr1bDQ?e=rKN1BC
- pretrained for MFNET: https://entuedu-my.sharepoint.com/:u:/g/personal/xuyu0014_e_ntu_edu_sg/EcbSBlB7PD5Bp_zPTuU84eQBwTFDX2HPdqjoCQ4dZn6Ajw?e=rgEbp2

#### 若有其他问题请及时通过邮件或者班主任与我联系
