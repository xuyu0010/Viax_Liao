# most of the code are from:
# https://github.com/bryanyzhu/two-stream-pytorch/blob/master/video_transforms.py
import cv2
import numpy as np

import torch

class Compose(object):
    """Composes several video_transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> video_transforms.Compose([
        >>>     video_transforms.CenterCrop(10),
        >>>     video_transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms, aug_seed=0):
        self.transforms = transforms
        for i, t in enumerate(self.transforms):
            t.set_random_state(seed=(aug_seed+i))   #set_random_state控制随机状态。

    def __call__(self, data):   # __call__一个类实例也可以变成一个可调用对象
        for t in self.transforms:
            data = t(data)
        return data

class Transform(object):
    """basse class for all transformation"""
    def set_random_state(self, seed=None):
        self.rng = np.random.RandomState(seed)
                        #numpy.random.RandomState()是一个伪随机数生成器。那么伪随机数是什么呢？
                        #伪随机数是用确定性的算法计算出来的似来自[0,1]均匀分布的随机数序列。并不真正的随机，但具有类似于随机数的统计特征，如均匀性、独立性等。


####################################
# Customized Transformations
####################################

class Normalize(Transform):     #归一化
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std    #归一化方法
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):    #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            t.sub_(m).div_(s)
        return tensor


class Resize(Transform):        #调整大小
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR    #INTER_LINEAR双线性插值（默认设置）
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size # [w, h]
        self.interpolation = interpolation

    def __call__(self, data):
        h, w, c = data.shape

        if isinstance(self.size, int):
            slen = self.size
            if min(w, h) == slen:
                return data     #data为原图
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]

        if (h != new_h) or (w != new_w):
            scaled_data = cv2.resize(data, (new_w, new_h), self.interpolation)  #resize缩放，(new_w, new_h)输出图像尺寸。
        else:
            scaled_data = data

        return scaled_data


class RandomScale(Transform):   #随机标度                        ？RandomScale(Transform)的思路？
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, make_square=False,                   # make_square生成正方形
                       aspect_ratio=[1.0, 1.0],             #aspect_ratio纵横比
                       slen=[224, 288],                #？
                       interpolation=cv2.INTER_LINEAR):     #插值
        assert slen[1] >= slen[0], \                        #判断
                "slen ({}) should be in increase order".format(scale)
        assert aspect_ratio[1] >= aspect_ratio[0], \
                "aspect_ratio ({}) should be in increase order".format(aspect_ratio)
        self.slen = slen # [min factor, max factor]
        self.aspect_ratio = aspect_ratio
        self.make_square = make_square
        self.interpolation = interpolation
        self.rng = np.random.RandomState(0)     #np.random.RandomState产生一个随机状态种子，该状态设置为0

    def __call__(self, data):
        h, w, c = data.shape
        new_w = w
        new_h = h if not self.make_square else w
        if self.aspect_ratio:
            random_aspect_ratio = self.rng.uniform(self.aspect_ratio[0], self.aspect_ratio[1])  #rng（随机数产生器），uniform函数可以返回指定范围的随机数
            if self.rng.rand() > 0.5:
                random_aspect_ratio = 1.0 / random_aspect_ratio
            new_w *= random_aspect_ratio
            new_h /= random_aspect_ratio
        resize_factor = self.rng.uniform(self.slen[0], self.slen[1]) / min(new_w, new_h)
        new_w *= resize_factor
        new_h *= resize_factor
        scaled_data = cv2.resize(data, (int(new_w+1), int(new_h+1)), self.interpolation)
        return scaled_data


class CenterCrop(Transform):        #适度裁剪
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):   #isinstance(实例对象，直接或间接类名、基本类型或者由它们组成的元组) 函数来判断一个对象是否是一个已知的类型
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))      #round() 方法返回浮点数x的四舍五入值。
        y1 = int(round((h - th) / 2.))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        return cropped_data

class RandomCrop(Transform):        #随机切割
    """Crops the given numpy array at the random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = self.rng.choice(range(w - tw))     #随机选取内容
        y1 = self.rng.choice(range(h - th))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        return cropped_data

class RandomHorizontalFlip(Transform):      #随机水平翻转
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """
    def __init__(self):
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        if self.rng.rand() < 0.5:
            data = np.fliplr(data)      #向左/向右翻转阵列
            data = np.ascontiguousarray(data)   #ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        return data

class RandomVerticalFlip(Transform):        #随机垂直翻转
    """Randomly vertically flips the given numpy array with a probability of 0.5
    """
    def __init__(self):
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        if self.rng.rand() < 0.5:
            data = np.flipud(data)      #上下翻转矩阵
            data = np.ascontiguousarray(data)
        return data

class RandomRGB(Transform):
    def __init__(self, vars=[10, 10, 10]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape

        random_vars = [int(round(self.rng.uniform(-x, x))) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape)
        for ic in range(0, c):
            var = random_vars[ic%base]
            augmented_data[:,:,ic] = np.minimum(np.maximum(data[:,:,ic] + var, 0), 255)
        return augmented_data

class RandomHLS(Transform):
    def __init__(self, vars=[15, 35, 25]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        assert c%3 == 0, "input channel = %d, illegal"%c

        random_vars = [int(round(self.rng.uniform(-x, x))) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape, )

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(data[:,:,3*i_im:(3*i_im+3)], cv2.COLOR_RGB2HLS)

        hls_limits = [180, 255, 255]
        for ic in range(0, c):
            var = random_vars[ic%base]
            limit = hls_limits[ic%base]
            augmented_data[:,:,ic] = np.minimum(np.maximum(augmented_data[:,:,ic] + var, 0), limit)

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(augmented_data[:,:,3*i_im:(3*i_im+3)].astype(np.uint8), \
                        cv2.COLOR_HLS2RGB)

        return augmented_data


class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # H, W, C = image.shape
            # handle numpy array
            image = torch.from_numpy(image.transpose((2, 0, 1)))    #图像转换
            # backward compatibility
            return image.float() / 255.0
