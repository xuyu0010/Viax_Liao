"""
Author: Yunpeng Chen
"""
import torch
import numpy as np

from .image_transforms import Compose, \            #构成
                              Transform, \          #格式转换
                              Normalize, \          #归一化
                              Resize, \             #调整大小
                              RandomScale, \
                              CenterCrop, \
                              RandomCrop, \
                              RandomHorizontalFlip, \
                              RandomRGB, \
                              RandomHLS


class ToTensor(Transform):      #数据类型转化，numpy的数组转化为torch的默认的tensor类型
    """Converts a numpy.ndarray (H x W x (T x C)) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3):
        self.dim = dim    #dim维度

    def __call__(self, clips):    #__call__()的作用是使实例能够像函数一样被调用，同时不影响实例本身的生命周期
        if isinstance(clips, np.ndarray):   #isinstance() 函数来判断一个对象是否是一个已知的类型
            H, W, _ = clips.shape
            # handle numpy array
            clips = torch.from_numpy(clips.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
                    #torch.from_numpy()：numpy中的ndarray转化成pytorch中的tensor 
                    #reshape（）重建数组，-1表示不知道，变换成三维的H*W的不知道多少个矩阵（方便之后用网络CNN处理）
                    #transpose交换多个维度，transpose只能对两个维度进行操作，permute没有限制
            # backward compatibility
            return clips.float() / 255.0
