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
