import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter

def get_hmdb51(data_root='./dataset/HMDB51',
               clip_length=8,
               train_interval=2,            #训练间隔
               val_interval=2,              #区间值
               mean=[0.485, 0.456, 0.406],  #均值
               std=[0.229, 0.224, 0.225],   #标准差
               seed=0,
               #seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,  #get_rank()返回当前进程组的排名
               **kwargs):                                                                       #is_initialized()检查是否已初始化默认进程组
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)
    #对数据进行归一化的操作；image = (image - mean) / std

    #随机采样    如何理解seed？
    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),    #导入的VideoIter
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_train.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([    #整合所有transforms
                                         transforms.RandomScale(make_square=True,   #？
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                                                            #RandomCrop：在一个随机的位置进行裁剪
                                         transforms.RandomHorizontalFlip(), #RandomHorizontalFlip：以0.5的概率水平翻转给定的PIL图像
                                         transforms.RandomHLS(vars=[15, 35, 25]),   #？
                                         transforms.ToTensor(), #ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )
    #顺序取样
    val_sampler   = sampler.SequentialSampling(num=clip_length,   
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'hmdb51_split1_test.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),   #Resize将输入PIL图像的大小调整为给定大小
                                         transforms.CenterCrop((224, 224)), #CenterCrop依据给定的size从中心裁剪 
                                         transforms.ToTensor(),           #ToTensor将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
                                         normalize,
                                      ]),
                      name='test',
                      )

    return (train, val)

def get_ucf101(data_root='./dataset/UCF101',
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=0,
               #seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
               **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    #对数据进行归一化的操作 
    normalize = transforms.Normalize(mean=mean, std=std)

    #随机采样
    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'trainlist01.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    #顺序取样
    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'testlist01.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )

    print(len(val))
    return (train, val)


def get_kinetics(data_root='./dataset/Kinetics',
                 clip_length=8,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=0,
                 #seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'train'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_train_avi.txt'),
                      # txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_train.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[256, 320]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_avi.txt'),
                      # txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         # transforms.Resize((256, 256)),
                                         transforms.RandomScale(make_square=False,
                                                                aspect_ratio=[1.0, 1.0],
                                                                slen=[256, 256]),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )
    return (train, val)

def get_aid11(data_root='./dataset/AID11',
               clip_length=8,
               train_interval=2,
               val_interval=2,
               mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225],
               seed=0,
               #seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
               **kwargs):
    """ data iter for ucf-101
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'AID11_split1_train.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=True,
                                                                aspect_ratio=[0.8, 1./0.8],
                                                                slen=[224, 288]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]),
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.SequentialSampling(num=clip_length,
                                               interval=val_interval,
                                               fix_cursor=True,
                                               shuffle=True)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'AID11_split1_test.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      )

    return (train, val)


def creat(name, batch_size, num_workers=16, **kwargs):

    if name.upper() == 'UCF101':    #upper() 返回一个字母全大写的字符串
        train, val = get_ucf101(**kwargs)
    elif name.upper() == 'HMDB51':
        train, val = get_hmdb51(**kwargs)
    elif name.upper() == 'KINETICS':
        train, val = get_kinetics(**kwargs)
    elif name.upper() == 'AID11':
        train, val = get_aid11(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))


    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=8,
        #batch_size=2*torch.cuda.device_count(), 
        shuffle=False,    #2*？
        num_workers=num_workers, pin_memory=False)
    # torch.utils.data.DataLoader该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
    #torch.cuda.device_count()返回可得到的GPU数量。
    return (train_loader, val_loader)
