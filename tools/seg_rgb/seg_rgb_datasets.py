import os
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import random
from PIL import Image


class SegmentDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = cfg['data_dir']
        self.is_trans = cfg['is_trans']
        self.is_train = cfg['is_train']
        self.data, self.label = self.read_image()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]

        if self.is_trans:
            image = self.trans_image(image)
        label = torch.LongTensor(label)

        sample = [image, label]
        return sample

    def read_image(self):
        x = []
        y = []
        img_list = os.listdir(self.data_dir)
        # img_list.sort()

        if self.is_train:
            for img in img_list:
                if 'new' not in img and 'mask' not in img:
                    image = Image.open(join(self.data_dir, img))
                    # arr = np.asarray(image, dtype=np.float32)
                    image = image.resize((self.cfg['image_size'][0], self.cfg['image_size'][1]))
                    x.append(image)

                    label_path = '{}_new.png'.format(img[:-4])
                    image = Image.open(join(self.data_dir, label_path))
                    arr = np.asarray(image, dtype=np.float32)
                    y.append(arr)

        else:
            img_list.sort()
            for img in img_list:
                image = Image.open(join(self.data_dir, img))
                # arr = np.asarray(image, dtype=np.float32)
                image = image.resize((self.cfg['image_size'][0], self.cfg['image_size'][1]))
                x.append(image)

                label = np.zeros((16, 16), dtype=np.float32)
                y.append(label)

        return x, y

    def trans_image(self, im):
        # im = im.resize((self.cfg['image_size'][0], self.cfg['image_size'][1]))
        trans = transforms.Compose([
            # transforms.Resize((self.cfg['image_size'][0], self.cfg['image_size'][1])),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg['mean'], self.cfg['std']),
        ])

        im_ = trans(im)
        # im_arr = im.numpy()

        return im_
