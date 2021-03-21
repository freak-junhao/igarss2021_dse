import sys

sys.path.append('../../')

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import random
from PIL import Image

from models import resnet
from efficientnet_pytorch import EfficientNet as ef_net
from efficientnet_pytorch.utils import efficientnet


def get_model(model_name, channels_num, classes_num=2, image_size=64):
    # resnext mode
    if 'resnext_50' in model_name:
        model = resnet.resnext50_32x4d(in_chs=channels_num, classes_num=classes_num)
    elif 'resnext_101' in model_name:
        model = resnet.resnext101_32x8d(in_chs=channels_num, classes_num=classes_num)

    # resnet mode
    elif 'resnet_34' in model_name:
        model = resnet.resnet34(in_chs=channels_num, classes_num=classes_num)
    elif 'resnet_50' in model_name:
        model = resnet.resnet50(in_chs=channels_num, classes_num=classes_num)
    elif 'resnet_101' in model_name:
        model = resnet.resnet101(in_chs=channels_num, classes_num=classes_num)

    elif 'wide_resnet50_2' in model_name:
        model = resnet.wide_resnet50_2(in_chs=channels_num, classes_num=classes_num)

    # efficientnet mode
    elif 'efficientnet_b4' in model_name:
        model = ef_net.from_name('efficientnet-b4', channels_num, image_size=image_size, num_classes=classes_num)
    elif 'efficientnet_b5' in model_name:
        model = ef_net.from_name('efficientnet-b5', channels_num, image_size=image_size, num_classes=classes_num)
    elif 'efficientnet_b7' in model_name:
        model = ef_net.from_name('efficientnet-b7', channels_num, image_size=image_size, num_classes=classes_num)
    else:
        # _, global_params = efficientnet(image_size=image_size, num_classes=classes_num)
        model = ef_net.from_name('efficientnet-b0', channels_num, image_size=image_size, num_classes=classes_num)
    return model


def open_file(txt_path):
    txt = open(txt_path)
    lines = txt.readlines()
    lines = [i.strip() for i in lines]
    txt.close()
    return lines


def combine_channels(channels_dir):
    image_lst = os.listdir(channels_dir)
    image_lst.sort()

    image = []
    for i in image_lst:
        channel = Image.open(os.path.join(channels_dir, i))
        channel = np.asarray(channel, dtype=np.float64)
        image.append(channel)
    return np.asarray(image, dtype=np.float64)


class ClassifiDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg['data_dir']
        self.is_trans = cfg['is_trans']
        self.txt_dir = cfg['txt_dir']
        self.data, self.label = self.read_txt()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]

        label = np.asarray(label, dtype=np.float32)
        label = torch.LongTensor(label)

        if self.is_trans:
            image = self.trans_channel(image)

        sample = [image, label]
        return sample

    def read_txt(self):
        images_list = open_file(self.txt_dir)
        x = []
        y = []
        for i in images_list:
            image_label = i.split(' ')
            data_path = os.path.join(self.data_dir, image_label[0])
            data = combine_channels(data_path)
            x.append(data)
            y.append(image_label[1])
        return x, y

    def trans_channel(self, arr):
        image = arr
        rand = random.randint(0, 1)
        image_new = []

        trans = transforms.Compose([
            transforms.Resize((self.cfg['image_size'][0], self.cfg['image_size'][1])),
            # transforms.RandomHorizontalFlip(rand),
            # transforms.RandomVerticalFlip(1 - rand),
        ])
        for i in range(len(image)):
            x = Image.fromarray(np.float64(image[i]))
            trans_x = trans(x)
            trans_x = np.asarray(trans_x, dtype=np.float32)
            image_new.append(trans_x)

        image_new = np.asarray(image_new, dtype=np.float32)
        image_new = torch.from_numpy(image_new)
        normalize_trans = transforms.Normalize(self.cfg['mean'], self.cfg['std'])
        image_new = normalize_trans(image_new)
        return image_new
