"""
Author: Yunpeng Chen
"""
import math
import numpy as np


#随机采样
class RandomSampling(object):       
    def __init__(self, num, interval=1, speed=[1.0, 1.0], seed=0):  #interval间隔
        assert num > 0, "at least sampling 1 frame"     #如果帧的总数>0,则至少采样1帧
        self.num = num
        self.interval = interval if type(interval) == list else [interval]      #type()取类型；如何interval的类型=列表
        self.speed = speed
        self.rng = np.random.RandomState(seed)      #numpy.random.RandomState()是一个伪随机数生成器

    def sampling(self, range_max, v_id=None, prev_failed=False):
        assert range_max > 0, \
            ValueError("range_max = {}".format(range_max))      #ValueError	传入无效的参数
        interval = self.rng.choice(self.interval)
        if self.num == 1:
            return [self.rng.choice(range(0, range_max))]       #随机选取
        # sampling
        speed_min = self.speed[0]       #min() 方法返回给定参数的最小值
        speed_max = min(self.speed[1], (range_max-1)/((self.num-1)*interval))       #？？此公式如何得出的？？
        if speed_max < speed_min:
            return [self.rng.choice(range(0, range_max))] * self.num                #？？？？
        random_interval = self.rng.uniform(speed_min, speed_max) * interval     #uniform() 方法将随机生成下一个实数
        frame_range = (self.num-1) * random_interval                                #（总帧数-1）*每帧的随机间隔=帧的范围
        clip_start = self.rng.uniform(0, (range_max-1) - frame_range)               #开始，公式怎么来的？
        clip_end = clip_start + frame_range
        return np.linspace(clip_start, clip_end, self.num).astype(dtype=np.int).tolist()
                #np.linspace在规定的时间内，返回固定间隔的数据。np. astype（dtype）创建新的数组；np.tolist()数组向列表的转换

#顺序采样
class SequentialSampling(object):
    def __init__(self, num, interval=1, shuffle=False, fix_cursor=False, seed=0):
        self.memory = {}
        self.num = num
        self.interval = interval if type(interval) == list else [interval]
        self.shuffle = shuffle      #排序
        self.fix_cursor = fix_cursor    #修复操作数据库？？
        self.rng = np.random.RandomState(seed)      #numpy.random.RandomState()是一个伪随机数生成器

    def sampling(self, range_max, v_id, prev_failed=False):
        assert range_max > 0, \
            ValueError("range_max = {}".format(range_max))
        num = self.num
        interval = self.rng.choice(self.interval)
        frame_range = (num - 1) * interval + 1      #帧的范围=（帧数-1）*间隔+1
        # sampling clips
        if v_id not in self.memory:
            clips = list(range(0, range_max-(frame_range-1), frame_range))
            if self.shuffle:
                self.rng.shuffle(clips)     #打乱数据集
            self.memory[v_id] = [-1, clips]
        # pickup a clip
        cursor, clips = self.memory[v_id]
        if not clips:
            return [self.rng.choice(range(0, range_max))] * num
        cursor = (cursor + 1) % len(clips)      #len() 方法返回对象（字符、列表、元组等）长度或项目个数。？？
        if prev_failed or not self.fix_cursor:
            self.memory[v_id][0] = cursor
        # sampling within clip
        idxs = range(clips[cursor], clips[cursor]+frame_range, interval)        #？
        return idxs


if __name__ == "__main__":      #__name__ 是当前模块名
#当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行。
    import logging
    logging.getLogger().setLevel(logging.DEBUG)     #logging.getLogger(name)方法进行初始化，setLevel：设置日志等级

    """ test RandomSampling() """

    random_sampler = RandomSampling(num=8, interval=2, speed=[0.5, 2])

    logging.info("RandomSampling(): range_max < num")           #这几句话的意思？range_max不同
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=2, v_id=1)))

    logging.info("RandomSampling(): range_max == num")
    for i in range(10):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=8, v_id=1)))

    logging.info("RandomSampling(): range_max > num")
    for i in range(90):
        logging.info("{:d}: {}".format(i, random_sampler.sampling(range_max=30, v_id=1)))


    """ test SequentialSampling() """
    sequential_sampler = SequentialSampling(num=3, interval=3, fix_cursor=False)

    logging.info("SequentialSampling():")
    for i in range(10):
        logging.info("{:d}: v_id = {}: {}".format(i, 0, list(sequential_sampler.sampling(range_max=14, v_id=0))))       #这句话的意思？
        # logging.info("{:d}: v_id = {}: {}".format(i, 1, sequential_sampler.sampling(range_max=9, v_id=1)))
        # logging.info("{:d}: v_id = {}: {}".format(i, 2, sequential_sampler.sampling(range_max=2, v_id=2)))
        # logging.info("{:d}: v_id = {}: {}".format(i, 3, sequential_sampler.sampling(range_max=3, v_id=3)))
