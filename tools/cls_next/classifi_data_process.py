from skimage.io import imread
from PIL import Image
import numpy as np
import random
import os
from os.path import join
import shutil
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


def select_channels(channel_lst, out_dir, in_dir, is_train=True):
    channels = read_txt(channel_lst)
    tile_list = os.listdir(in_dir)
    outpath_name = 'Train/' if is_train else 'Val'

    for tile in tile_list:
        pic_list = os.listdir(join(in_dir, tile))
        index_list = [pic_list.index(y) for y in channels]
        target_tile_path = join(out_dir, outpath_name, tile)
        make_dirs(target_tile_path)

        for mv in index_list:
            old_file = join(in_dir, tile, pic_list[mv])
            new_file = join(out_dir, outpath_name, tile, pic_list[mv])
            if os.path.exists(new_file) is not True:
                shutil.copyfile(old_file, new_file)

        if is_train:
            gt_name = "groundTruth.tif"
            shutil.copyfile(join(in_dir, tile, gt_name), join(out_dir, outpath_name, tile, gt_name))


def mean_std(channel_lst, train_dir, val_dir):
    channels = read_txt(channel_lst)
    class_list = [
        train_dir, val_dir
    ]

    all_mean = []
    all_std = []
    print("Start compute mean : ")
    for i in tqdm.tqdm(channels):
        channel_mean = 0
        pixel_num = 0

        for t_or_v in class_list:
            tile_lst = os.listdir(t_or_v)
            pixel_num += len(tile_lst)

            for tile in tile_lst:
                image_path = join(t_or_v, tile, i)
                image = imread(image_path)
                channel_mean += np.sum(image)

        pixel_num *= 800 * 800
        mean = channel_mean / pixel_num
        all_mean.append(mean)
    mean = np.asarray(all_mean, dtype=np.float32)
    np.savetxt('./txt_files/mean.txt', mean, fmt="%.4f")

    print("Start compute std : ")
    for i in tqdm.trange(len(channels)):
        channel_std = 0
        pixel_num = 0

        for t_or_v in class_list:
            tile_lst = os.listdir(t_or_v)
            pixel_num += len(tile_lst)

            for tile in tile_lst:
                image_path = join(t_or_v, tile, channels[i])
                image = imread(image_path)
                _mean = all_mean[i]
                channel_std += np.sum((image - _mean) ** 2)

        pixel_num *= 800 * 800
        std = np.sqrt(channel_std / pixel_num)
        all_std.append(std)
    all_std = np.asarray(all_std, dtype=np.float32)
    np.savetxt('./txt_files/std.txt', all_std, fmt="%.4f")


def pic_cut(out_dir, in_dir, cut_mode=None, is_train=True, class_num=2):
    if cut_mode is None:
        cut_mode = [1, 3]

    patch_num = np.zeros(class_num, dtype=np.uint16)
    tile_list = os.listdir(in_dir)

    count = 1
    for tile in tile_list:
        ch_lst = os.listdir(in_dir + tile)  # channels_list
        if is_train:
            print(tile + ' {}-train: '.format(count))
            ch_lst.remove("groundTruth.tif")
        else:
            print(tile + ' {}-test: '.format(count))
        count += 1

        full_tif = []  # multiple channels
        for channel in ch_lst:
            read_tif = Image.open(join(in_dir, tile, channel))
            read_tif = np.asarray(read_tif, dtype=np.float64)
            full_tif.append(read_tif)
        full_tif = np.asarray(full_tif, dtype=np.float64)

        gt = ''  # ground truth
        if is_train:
            gt = Image.open(join(in_dir, tile, 'groundTruth.tif'))
            gt = np.asarray(gt, dtype=np.uint8)
            if class_num < 4:
                gt = transform_label(gt, cut_mode)
                gt_new = Image.fromarray(np.uint8(gt))
                gt_new.save(join(in_dir, tile, 'groundTruth_new.tif'))
        else:
            patch_num[0] = 0

        for width in tqdm.trange(16):
            for height in range(16):
                w_start = int(50 * width)
                h_start = int(50 * height)
                patch = full_tif[:, w_start:w_start + 50, h_start:h_start + 50]

                if is_train:
                    label_rgb = gt[width][height]
                    patch_num[label_rgb] += 1
                    save_path = join(out_dir[0], out_dir[label_rgb + 1], "patch_{:0>4d}/".format(patch_num[label_rgb]))
                else:
                    patch_num[0] += 1
                    save_path = join(out_dir, tile, "patch_{:0>3d}/".format(patch_num[0]))

                make_dirs(save_path)
                channel_size, _, _ = patch.shape
                for i in range(channel_size):
                    patch_save = patch[i]
                    patch_save = Image.fromarray(np.float64(patch_save))
                    patch_save.save(join(save_path, ch_lst[i]))


def channel_aver(name_list, out_dir, in_dir, is_train=True):
    combined_name = name_list[0][-7:]
    outpath_name = 'Train/' if is_train else 'Val'

    tile_list = os.listdir(in_dir)

    for tile in tile_list:
        target_tile_path = join(out_dir, outpath_name, tile)
        make_dirs(target_tile_path)

        array_temp = np.zeros((800, 800))
        for pic in name_list:
            read_tif = Image.open(join(in_dir, tile, pic))
            read_tif = np.asarray(read_tif, dtype=np.float64)
            array_temp += read_tif
        array_temp /= len(name_list)

        save_path = join(out_dir, outpath_name, tile, combined_name)
        saved = Image.fromarray(np.float64(array_temp))
        saved.save(save_path)


def create_train_list(out_dir, in_dir):
    class_list = os.listdir(in_dir)
    train = []
    for cls in class_list:
        cls_num = 1 - class_list.index(cls)
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


def trans_to_image(result_lst, tile_dir, out_dir):
    tile_order = os.listdir(tile_dir)
    count = 0

    for k in tile_order:  # 19
        image = []
        for i in range(16):
            row = []
            for j in range(16):
                result_cls = result_lst[count]
                count += 1
                row.append(result_cls)

            image.append(row)
        image_save = np.asarray(image, dtype=np.uint8)
        image_save = Image.fromarray(np.uint8(image_save))
        image_save.save(join(out_dir, "{}.tif".format(k[4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on IEEE GRSS')
    parser.add_argument('select_txt', type=str, help='Selected channels txt.')

    parser.add_argument('--classifi_mode', type=int, default=0, help='Classify mode choice.')
    args = parser.parse_args()

    data_root = "/root/data/cls/"
    classifi_path = join(data_root, "classifi_data/")
    data_path = [join(data_root, "Train/"), join(data_root, "Val/")]

    selected_txt = "./txt_files/{}.txt".format(args.select_txt)
    select_channels(selected_txt, classifi_path, data_path[0])
    select_channels(selected_txt, classifi_path, data_path[1], is_train=False)

    """average_path = "./txt_files/average_list.txt"
    average_lst = read_txt(average_path)
    average_path = join(classifi_path, "Average/")
    channel_aver(average_lst, average_path, data_path[0])
    channel_aver(average_lst, average_path, data_path[1], is_train=False)
    """

    """
    train_cut_path = ["../../data/classifi_data/cut_train/",
                      "H_without_E/",
                      "NoH_without_E/",
                      "H_with_E/",
                      "NoH_with_E/"]
    """

    train_cut_path = [
        ["/root/data/cls/classifi_data/cut_train/",
         "NoH/",
         "H/"],
        ["/root/data/cls/classifi_data/cut_train/",
         "E/",
         "NoE/"],
        ["/root/data/cls/classifi_data/cut_train/",
         "Others",
         "H_without_E/"],
    ]

    set_one = [
        [1, 3],
        [1, 2],
        [1],
    ]

    train_data_path = "/root/data/cls/classifi_data/Train/"
    pic_cut(train_cut_path[args.classifi_mode], train_data_path, cut_mode=set_one[args.classifi_mode])

    val_cut_path = "/root/data/cls/classifi_data/cut_val"
    val_data_path = "/root/data/cls/classifi_data/Val/"
    pic_cut(val_cut_path, val_data_path, is_train=False)

    generate_list(classifi_path, [train_cut_path[args.classifi_mode][0], val_cut_path])
    mean_std("./txt_files/{}.txt".format(args.select_txt), train_data_path, val_data_path)
