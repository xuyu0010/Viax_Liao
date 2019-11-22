# General FYP Action Recognition Framework

This repository is the framework for Action Recognition for FYP 2020.

## Implementation

This framework is in PyTorch

### Normalization
The inputs are substrated by mean RGB = [ 124, 117, 104 ], and then multiplied by 0.0167.


## Usage

Train motion from scratch:
```
python train_kinetics.py
```

Fine-tune with pre-trained model:
```
python train_ucf101.py
```
or 
```
python train_hmdb51.py
```

Evaluate the trained model:
```
cd test
# the default setting is to test trained model on ucf-101 (split1)
python evaluate_video.py
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
- 主程序中目前只有train_ucf101.py换成了CPU版本，若需要使用train_hmdb51.py（使用HMDB51数据集）请参照train_ucf101.py进行修改。

### 数据库下载链接如下（可在中国大陆使用）
- UCF101：https://entuedu-my.sharepoint.com/:u:/g/personal/xuyu0014_e_ntu_edu_sg/EaDYMqmoSPNHqM3sVyEH4jkBvPJP6fGDYBTX6n2vB0-jXA?e=I3OxfT
- HMDB51：https://entuedu-my.sharepoint.com/:u:/g/personal/xuyu0014_e_ntu_edu_sg/ETkMjaDh3DdCjaJ-PfX5tW8B043TZH26aeheP4laCr1bDQ?e=rKN1BC

#### 若有其他问题请及时通过邮件或者班主任与我联系
