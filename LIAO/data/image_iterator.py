"""
Author: Yunpeng Chen
"""
import os
import cv2
import numpy as np

import torch.utils.data as data     #实现自由的数据读取
import logging

#class torch.utils.data.Dataset：作用: (1) 创建数据集,有__getitem__(self, index)函数来根据索引序号获取图片和标签, 有__len__(self)函数来获取数据集的长度.


class ImageListIter(data.Dataset):      #torch.utils.data.Dataset是代表自定义数据集方法的抽象类
                 #ImageListIter——子类；data.Dataset——基类；
    
    def __init__(self, 
                 image_prefix,
                 txt_list,
                 image_transform,
                 name="",
                 force_color=True):
        super(ImageListIter, self).__init__()
                # super的内核：mro（method resolution order），表示了类继承体系中的成员解析顺序。
                #super(Leaf, self).__init__()的意思是说：
                #1.获取self所属类的mro, 也就是[Leaf, Medium1, Medium2, Base]
                #2.从mro中Leaf右边的一个类开始，依次寻找__init__函数。这里是从Medium1开始寻找
                #3.一旦找到，就把找到的__init__函数绑定到self对象，并返回
            
        # load image list
        self.image_list = self._get_video_list(txt_list=txt_list)   #得到视频播放列表给图像列表
                

        # load params
        self.force_color = force_color
        self.image_prefix = image_prefix
        self.image_transform = image_transform
        logging.info("ImageListIter ({:s}) initialized, num: {:d})".format(name,
                      len(self.image_list)))
            #logging.info函数, 表示: 打印(记录)INFO级别的日志信息
        
    def get_image(self, index):     #获取当前视频信息
        # get current video info
        im_id, label, img_subpath = self.image_list[index]

        # load image
        image_path = os.path.join(self.image_prefix, img_subpath)
        if self.force_color:
            cv_read_flag = cv2.IMREAD_COLOR         #加载一张彩色图片,忽视它的透明度。
        else:
            cv_read_flag = cv2.IMREAD_GRAYSCALE     #加载一张灰度图
        cv_img = cv2.imread(image_path, cv_read_flag)   #cv2.imread(路径，读取方式)函数读取图像；读进来直接是BGR 格式数据格式在 0~255
        image_input = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)   #cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
                                                                #cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
                                                              
        # apply image augmentation
        if self.image_transform is not None:        #图像增强
            image_input = self.image_transform(image_input)
        return image_input, label, img_subpath

#如果你希望定制的容器是不可变的话，你只需要定义__len__()和__getitem__这两个魔法方法。

    def __getitem__(self, index):       #__getitem__(self)定义获取容器中指定元素的行为，相当于self[key]，即允许类对象可以有索引操作。
        image_input, label, img_subpath = self.get_image(index)
        return image_input, label


    def __len__(self):      #__len__(self) 定义当被len()函数调用时的行为（返回容器中元素的个数）
        return len(self.image_list)


    def _get_video_list(self, txt_list):
        # formate:
        # [im_id, label, image_subpath]
        assert os.path.exists(txt_list), "Failed to locate: {}".format(txt_list)
            #assert条件判断，当这个关键字后边的条件为假的时候，程序自动崩溃并抛出AssertionError的异常。
            #一般来说我们可以用assert在程序中置入检查点，当需要确保程序中某个条件一定为真才能让程序正常工作的话，assert关键字就非常有用了。
            #os.path.exists()方法可以直接判断文件/文件夹是否存在
            
        
        # building dataset
        logging.info("Building dataset ...")
        image_list = []
        with open(txt_list) as f:       #文件读取
            lines = f.read().splitlines()   #读取文本文件的行数据
            logging.info("Found {} images in '{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):    #对于列表，既要遍历索引又要遍历元素时
                im_id, label, image_subpath = line.split()    #split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
                info = [int(im_id), int(label), image_subpath]
                image_list.append(info) #append()方法向列表末尾添加新的对象（元素）。

        return image_list
