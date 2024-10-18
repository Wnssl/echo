import csv
import os
import random

import cv2
import numpy as np
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from collections import OrderedDict
from losses import *

transform = transforms.Compose([
    transforms.Resize((112, 112)),
])

DEFAULT_MEAN = np.array([0.12741163, 0.1279413 , 0.12912785]) * 255
DEFAULT_STD  = np.array([0.19557191, 0.19562256, 0.1965878]) * 255


# from preprocessing import normalize_frames

from utils import get_lds_kernel_window
from scipy.ndimage import convolve1d

# 读取avi视频

class EchoData(Dataset):

    def __init__(self, file_path,
                 isTrain=True,
                 process_type='a',
                 input_dim_transpose=False,
                 class_nums=7,
                 codebook_path='codebook.csv',
                 reweight='sqrt_inv',
                 lds=True,
                 lds_kernel='gaussian',
                 lds_ks=5, lds_sigma=2,
                 pad=8, frames=32, frequency=2,
                 use_mask=True,
                 mean=DEFAULT_MEAN,
                 std=DEFAULT_STD,
                 ):
        """读取训练的数据集"""
        # 保存需要训练的数据样本 和标签
        self.datas = []
        self.targets = []
        self.weights = []

        # pad 值     下采样帧数     采样周
        self.pad = pad
        self.frames = frames
        self.frequency = frequency

        #std  mean
        self.mean = mean
        self.std = std

        #use_mask 是否使用掩码
        self.use_mask = use_mask

        # 判断是否为train
        self.train = isTrain

        # 输入数据维度  (batch, channel, time, weight, height)
        # 判断数据是否需要转置 转为 (batch, time, channel ,weight, height)
        self.input_dim_transpose = input_dim_transpose

        # 处理数据平衡策略
        # a -- 默认不处理 b -- re-sampling c -- re-weighting
        self.pt = process_type

        # 所有射血分数分类 类别数
        self.class_nums = class_nums

        # codebook path
        self.codebook_path = codebook_path

        # 训练文件 路径
        self.file_path = file_path

        # 对应训练文件路径下的所有文件
        self.files = []
        # 读取文件
        files = os.listdir(self.file_path)
        # 将读取到的file文件名单独分割出来 同时保存不包含后缀的文件名
        for f in files:
            file_name = f.split('.')[0]
            self.files.append(file_name)

        # 读取codebook
        reader = read_csv(self.codebook_path)

        # 创建一个空字典
        self.dict = {}

        # 生成     file : rvef        dict
        for line in reader:
            if line[0] in self.files:
                rvef = line[-2]
                rvef = float(rvef) / 100
                rvef = float(rvef)
                self.dict[line[0]] = rvef

        # 对字典进行排序
        sorted_dict = dict(sorted(self.dict.items(), key=lambda x: x[1]))
        self.dict = sorted_dict

        # 计算min_rvef 和 max_rvef
        min_rvef = min(self.dict.values())
        max_rvef = max(self.dict.values())

        # 计算按照类别数 划分的step
        step = (max_rvef - min_rvef) / self.class_nums
        # print(step)

        # 划分区间 生成区间数量
        data = np.array(list(self.dict.values()))
        hist, bins = np.histogram(data, bins=self.class_nums, range=(min_rvef, max_rvef))

        if isTrain:
            if self.pt == 'a':
                print(f' 不适用数据平衡方法 ')
                for k, v in self.dict.items():
                    self.datas.append(k)
                    self.targets.append(v)
            elif self.pt == 'b':
                print(f'使用re-sampling平衡')
                # re-sampling方法
                # 打印每个区间的元素数量
                # for i in range(len(hist)):
                #     print(f"区间 {bins[i]} - {bins[i + 1]}: {hist[i]} 个元素")

                re_sampling_num = min(hist) * 10
                print(f'重采样个数{re_sampling_num}')
                index = 0
                for i in range(self.class_nums):
                    print(f'当前类别:{i}')
                    print(f'区间 {bins[i]} - {bins[i + 1]} : {hist[i]} 个元素')
                    print(f'当前索引{index}')
                    # 获取当前类别区间的文件
                    # 根据self.dict有序
                    # 当前区域files, targets
                    files, targets = [], []
                    for k in list(self.dict.keys())[index: index + hist[i]]:
                        files.append(k)
                        targets.append(self.dict[k])
                    # 验证target 加载是否正确
                    for v in targets:
                        if bins[i + 1] >= v >= bins[i]:
                            pass
                        else:
                            print(v)
                            assert 1 < 0
                    print(f"当前区域文件数量：{len(files)}")
                    # 索引更新
                    index += hist[i]
                    # 对不同长度的files做不同的变换
                    if len(files) > re_sampling_num:
                        # down - sampling
                        sample_files = random.sample(files, re_sampling_num)

                        for f in sample_files:
                            self.datas.append(f)
                            self.targets.append(self.dict[f])

                    else:

                        ratio = re_sampling_num // len(files)
                        diff = re_sampling_num % len(files)

                        sample_files = files * ratio

                        temp_files = random.sample(files, diff)

                        for f in temp_files:
                            sample_files.append(f)

                        for f in sample_files:
                            self.datas.append(f)
                            self.targets.append(self.dict[f])

                    assert len(sample_files) == re_sampling_num
                    # 验证添加的 target 是否正确
                    for t in self.targets[i * re_sampling_num: (i + 1) * re_sampling_num]:
                        if bins[i] <= t <= bins[i + 1]:
                            pass
                        else:
                            print('data error')
                            assert 1 < 0

            elif self.pt == 'c':
                # 计算权重
                # re-weightig方法
                print('使用re-weighting方法')
                # 2024 6 12 修改 不再使用re-sampling的分类来计算 weight
                weight_dict = {}

                weights = self._weights(reweight, lds, lds_kernel, lds_ks, lds_sigma)

                for k, v in self.dict.items():
                    self.datas.append(k)
                    self.targets.append(v)
                    # 计算类别   输入dict 对应的target  --> cls_id   int
                    cls = round(v * 100) - 22
                    self.weights.append(weights[cls])

            else:
                print(' input process type error')
                assert 1 < 0
        else:
            for k, v in self.dict.items():
                self.datas.append(k)
                self.targets.append(v)

    def __getitem__(self, item):
        """在此处实现对数据集的加载"""
        file = self.datas[item]
        target = self.targets[item]
        # print(target)
        path = self.file_path + '/' + file + '.avi'

        video = self.load_video(path).astype(np.float32)

        video = video.transpose(3, 0, 1, 2)
        video = self.sample_video(video)

        if self.pad:
            video = self.pad_video(video)

        video = self.normalize_video(video)

        video_tensor = torch.from_numpy(video)

        # if not self.use_mask:
        #     mask = video_tensor[0, :, :, :]
        #     images = video_tensor[1, :, :, :]
        #     video_tensor = torch.stack([images, images, images], dim=0)
        #     print(video_tensor.shape)
        # 正常的维度 是 (b, t, s, w, h) -> (b, s, t, w, h)
        if self.input_dim_transpose:
            video_tensor = torch.transpose(video_tensor, 0, 1)
        # print(target)
        target = torch.tensor(target)
        # 数据增强只对训练集有效
        if self.train:
            # 判断数据是否要经过数据增强
            enhance = random.choice([True, False])
            if enhance:
                video_tensor = rotate(video_tensor)
                video_tensor = flip(video_tensor)

        # resize
        video_tensor = transform(video_tensor)
        # 判断是否需要加权 只对训练集有效
        if self.pt == 'c' and self.train:
            weight = self.weights[item]
            return video_tensor,  target, weight
        else:
            return video_tensor, target

    def __len__(self):
        """返回训练集的长度"""
        return len(self.datas)

    def _get_class(self, target) -> int:
        # 获取对象应该属于 类别
        data = np.array(list(self.dict.values()))
        min_rvef = min(data)
        max_rvef = max(data)

        hist, bins = np.histogram(data, bins=self.class_nums, range=(min_rvef, max_rvef))
        for i in range(len(hist)):
            if bins[i + 1] > target >= bins[i]:
                # print(f"区间 {bins[i]} - {bins[i + 1]}: {hist[i]} 个元素 target:   {target} ")
                break
        # 返回类别号
        return i

    def _weights(self, reweight='inverse', lds=True, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, re_cls=None,
                 re_cls_type=None) -> list:
        '''  计算各个类别的权重  返回值是各个类别对应的权重 '''
        # 判断reweight的类型
        print('正在计算权重')

        assert reweight in {'None', 'inverse', 'sqrt_inv'}
        # 数据
        data = np.array(list(self.dict.values()))
        # show_hist(data)
        # 对射血分数四舍五入进行统计， 每个数值有多少个样本
        # 先初始化
        value_dict = {x: 0 for x in range(round(min(data) * 100), round(max(data) * 100) + 1)}
        print('初始化label -num 字典')
        # print(value_dict)
        for d in data:
            try:
                value_dict[round(d * 100)] += 1
            except Exception as e:
                print(e)
                print('error label')
                print(round(d * 100))

        # print(value_dict)
        # show_dict(value_dict)

        print(f"Using re-weighting: [{reweight.upper()}]")

        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight

        num_per_label = list(value_dict.values())
        # print(num_per_label)

        if not len(num_per_label):
            print('data error')
            return None

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray(num_per_label), weights=lds_kernel_window, mode='constant')
            print('成功进行平滑')
            i = 0
            for k, v in value_dict.items():
                value_dict[k] = smoothed_value[i]
                i += 1
            # 决定是否要对smooth_value进行分类
            if not re_cls:
                # 不对平滑结果再次分类
                print(f're_cls : {re_cls}')
            else:
                # 对value_dict 进一步分类
                pass
        # 对values 进一步处理， 如果出现0值 则使用一个极大值进行替代，倒数很小， 降低影响
        for k, v in value_dict.items():
            if v == 0:
                value_dict[k] = 99999999

        print(value_dict)
        weights = [np.float32(1 / x) for x in value_dict.values()]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        # print(weights)
        # print(len(weights))
        return weights

    def get_bucket_info(self, max_target=76, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        value_dict = {x: 0 for x in range(20, max_target)}
        labels = self.targets
        for label in labels:
            if int(label) < max_target:
                value_dict[int(label)] += 1
        bucket_centers = np.asarray([k for k, _ in value_dict.items()])
        bucket_weights = np.asarray([v for _, v in value_dict.items()])
        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            bucket_weights = convolve1d(bucket_weights, weights=lds_kernel_window, mode='constant')

        bucket_centers = np.asarray([bucket_centers[k] for k, v in enumerate(bucket_weights) if v > 0])
        bucket_weights = np.asarray([bucket_weights[k] for k, v in enumerate(bucket_weights) if v > 0])
        bucket_weights = bucket_weights / bucket_weights.sum()
        return bucket_centers, bucket_weights


    def load_video(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        capture = cv2.VideoCapture(path)

        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video = np.zeros((count, height, width, 3), np.uint8)

        for i in range(count):
            out, frame = capture.read()
            if not out:
                raise ValueError("Problem when reading frame #{} of {}.".format(i, path))
            # print(frame.shape)
            video[i, :, :, :] = frame

        # cv2.imshow('1,', video[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return video

    def pad_video(self, video):
        """ 对video进行padding """
        """ 同时包含对video 的randomcrop"""
        if not self.pad:
            return video

        c, t, h, w = video.shape

        tvideo = np.zeros((c, t, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
        tvideo[:, :, self.pad:- self.pad, self.pad:- self.pad] = video  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * self.pad, 2)

        video_temp = tvideo[:, :, i:(i + h), j:(j + w)]
        # print(tvideo.shape)
        # print(video_temp.shape)
        return video_temp

    def normalize_video(self, video):
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        return video

    def sample_video(self, video):
        c, f, h, w = video.shape
        frames = self.frames

        # 帧数不足， 进行补全
        if f < frames * self.frequency:
            video = np.concatenate((video, np.zeros((c, frames * self.frequency - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape

        start = np.random.choice(f - (frames - 1) * self.frequency, 1)
        # print(start)
        temp = start + self.frequency * np.arange(frames)
        video = tuple(video[:, s + self.frequency * np.arange(frames), :, :] for s in start)
        video = video[0]

        return video


def read_csv(codebook):
    ''' 读取csv 文件'''
    reader = csv.reader(open(codebook, 'r+'))
    next(reader)
    return reader


def rotate(tensor):
    """对tensor进行旋转"""

    # 随机角度， 进行旋转
    a = random.randint(-30, 30)
    images = []
    for image in tensor:
        rotate_image = TF.rotate(image, a)
        images.append(rotate_image)
    tensor1 = torch.stack(images)

    return tensor1


def flip(tensor):
    """对tensor进行水平翻转"""
    flag = random.choice([0, 1])
    horizontal_flip = transforms.RandomHorizontalFlip(p=1)
    if flag:
        return tensor
    else:
        images = []
        for image in tensor:
            flip_image = horizontal_flip(image)
            images.append(flip_image)
        tensor1 = torch.stack(images)

    return tensor1


def cv_show(tensor):
    """ 可视化预处理tensor """
    array = tensor.numpy()
    array = array.transpose(0, 2, 3, 1)
    count = 0
    for j in array:
        plt.imshow(j[:, :, 1], cmap='gray')
        plt.savefig('figs/savefig_' + str(count) + '.png')
        plt.show()
        count += 1


def show_hist(data, output_path='/'):
    ''' 绘制直方图 '''

    plt.figure()
    plt.hist(data, bins=70, color='skyblue', edgecolor='black')
    # 添加标题和标签
    plt.title('Histogram of ALL RVEF DATA')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # 保存图像
    # plt.savefig(os.path.join(output_path, 'result1.png'))
    # 显示图形
    plt.show()


def show_dict(data):
    """ 绘制 数值字典的分布"""
    import matplotlib.pyplot as plt

    # 提取字典的键和值
    labels = list(data.keys())
    values = list(data.values())

    # 创建一个条形图
    plt.bar(labels, values)

    # 添加标题和标签
    plt.title('Bar Chart of Numerical Dictionary')
    plt.xlabel('Keys')
    plt.ylabel('Values')

    # 显示图形
    plt.show()


# #def __init__(self, file_path,
#                  isTrain=True,
#                  process_type='b',
#                  class_nums=7,
#                  codebook_path='codebook.csv',
#                  reweight='sqrt_inv',
#                  lds=True,
#                  lds_kernel='gaussian',
#                  lds_ks=5, lds_sigma=2
#                  ):
#


class datasetRnc(EchoData):
    # Rnc loss
    def __getitem__(self, index):
        view1 = super().__getitem__(index)
        view2 = super().__getitem__(index)

        return view1, view2


# 快速接口
def train_data(file_path, isTrain=True, process_type='c',
               input_dim_transpose=True,
               class_nums=7, codebook_path='codebook.csv',
               reweight='sqrt_inv', lds=True, lds_kernel='gaussian',
               lds_ks=5, lds_sigma=2
               ):
    ''' 快速生成 train dataset的方法'''
    train = EchoData(file_path, isTrain, process_type, input_dim_transpose,
                     class_nums, codebook_path, reweight,
                     lds, lds_kernel, lds_ks, lds_sigma)
    return train


def test_data(file_path, isTrain=False, process_type='c', input_dim_transpose=True,
              class_nums=7, codebook_path='codebook.csv',
              reweight='sqrt_inv', lds=True, lds_kernel='gaussian',
              lds_ks=5, lds_sigma=2
              ):
    ''' 快速生成 train dataset的方法'''
    test = EchoData(file_path, isTrain, process_type, input_dim_transpose,
                    class_nums, codebook_path, reweight,
                    lds, lds_kernel, lds_ks, lds_sigma)
    return test


def train_data_rnc(file_path, isTrain=True, process_type='c',
                   input_dim_transpose=True,
                   class_nums=7, codebook_path='codebook.csv',
                   reweight='sqrt_inv', lds=True, lds_kernel='gaussian',
                   lds_ks=5, lds_sigma=2):
    train = datasetRnc(file_path, isTrain, process_type, input_dim_transpose,
                       class_nums, codebook_path, reweight,
                       lds, lds_kernel, lds_ks, lds_sigma)

    return train


if __name__ == '__main__':
    train = train_data('../train_video')
    print(len(train))
    #     print(len(train))
    #     # print(train[3569])
    # data, target, weight = train[0]
    from torch.utils.data import random_split
    train ,validate = random_split(train, [len(train)-400, 400])
    
    data = validate[100]
    print(data)

    # print(weight)
    # print(target)
    # # print(data.shape)
    # print(data[0][0])
    # for i in data[0][0]:
    #     for j in i:
    #         print(j)
    #     print('\n')

    #     print(target.shape)
    #     # print(target)
    # tensor, target, weight = train[100]
#     print(tensor.shape)
#     cv_show(tensor)
#     tensor = rotate(tensor)
# cv_show(tensor)
# tensor = flip(tensor)
# cv_show(tensor)
