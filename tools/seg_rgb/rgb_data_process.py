from PIL import Image
import numpy as np
import random
import os
from os.path import join
import tqdm
import argparse


def make_dirs(path):
    if os.path.exists(path) is not True:
        os.makedirs(path)


def read_txt(txt_dir):
    txt_list = open(txt_dir)
    txt = txt_list.readlines()
    txt = [i.strip() for i in txt]
    txt_list.close()
    return txt


def change_num(arr, index, num):
    for x, y in index:
        arr[x][y] = num
    return arr


def transform_label(label, interest_class):
    x, y = label.shape

    label_new = np.zeros((x, y), dtype=np.uint8)
    for i in interest_class:
        index = np.argwhere(label == i)
        label_new = change_num(label_new, index, 1)

    return label_new


def mean_std(train_dir, val_dir):
    # pic_name = rgb_name
    class_list = [
        train_dir, val_dir
    ]

    print("Start computing mean and std --> ")
    mean = np.zeros(3)
    pixel_num = 0.0
    for t_or_v in class_list:
        image_lst = os.listdir(t_or_v)

        for image in image_lst:
            if 'new' not in image and 'mask' not in image:
                pixel_num += 1
                image_path = join(t_or_v, image)
                image = Image.open(image_path)
                r, g, b = image.split()

                mean[0] += np.sum(r)
                mean[1] += np.sum(g)
                mean[2] += np.sum(b)
                # mean[3] += np.sum(a)

    pixel_num *= 800 * 800
    mean = mean / pixel_num
    np.savetxt('./mean.txt', mean, fmt="%.4f")

    std = np.zeros(3)
    pixel_num = 0.0
    for t_or_v in class_list:
        image_lst = os.listdir(t_or_v)

        for image in image_lst:
            if 'new' not in image and 'mask' not in image:
                pixel_num += 1
                image_path = join(t_or_v, image)
                image = Image.open(image_path)
                r, g, b = image.split()

                std[0] += np.sum((r - mean[0]) ** 2)
                std[1] += np.sum((g - mean[1]) ** 2)
                std[2] += np.sum((b - mean[2]) ** 2)
                # std[3] += np.sum((a - mean[0]) ** 2)

    pixel_num *= 800 * 800
    std = np.sqrt(std / pixel_num)
    np.savetxt('./std.txt', std, fmt="%.4f")


def create_train_list(out_dir, in_dir):
    class_list = os.listdir(in_dir)
    train = []
    for cls in class_list:
        cls_num = class_list.index(cls)
        pic_list = os.listdir(join(in_dir, cls))
        random.shuffle(pic_list)

        train_list = ["{}/{} {}".format(cls, i, cls_num) for i in pic_list]
        train.extend(train_list)

    random.shuffle(train)
    train = np.asarray(train)
    np.savetxt(join(out_dir, "train.txt"), train, fmt='%s')


def create_val_list(out_dir, in_dir):
    tile_list = os.listdir(in_dir)
    val = []
    for tile in tile_list:
        patch = os.listdir(join(in_dir, tile))
        patch = ["{}/{} 0".format(tile, i) for i in patch]
        patch.sort()

        val.extend(patch)
    val = np.asarray(val)
    np.savetxt(join(out_dir, "val.txt"), val, fmt='%s')


def generate_list(output_path, input_path):
    create_train_list(output_path, input_path[0])
    create_val_list(output_path, input_path[1])


def new_label(in_dir, label_mode=None):
    if label_mode is None:
        label_mode = [1, 3]

    pic_list = os.listdir(in_dir)
    pic_list.sort()
    for pic in pic_list:
        if 'mask' in pic:
            gt = Image.open(join(in_dir, pic))
            gt = np.asarray(gt, dtype=np.uint8)

            gt_new = transform_label(gt, label_mode)
            gt_new = Image.fromarray(np.uint8(gt_new))
            gt_new.save(join(in_dir, '{}_new.png'.format(pic[:-9])))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Trains rgb on IEEE GRSS')
    # parser.add_argument('classifi_mode', type=int, help='Classify mode choice.')
    # args = parser.parse_args()

    data_root = "../../../data/seg_rgb/train"
    new_label(data_root)
    mean_std("../../../data/seg_rgb/train", "../../../data/seg_rgb/val/")
