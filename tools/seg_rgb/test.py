import sys

sys.path.append('../../')

import argparse
import os

import torch
from torch.utils.data import DataLoader

import tqdm
import numpy as np
from PIL import Image
from models.unet import UNet
from models.smp_models import SmpModel16
from tools.seg_rgb.seg_rgb_datasets import SegmentDataset


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains Unet on IEEE GRSS')
    # Positional arguments
    parser.add_argument('model_name', type=str, help='Model name.')
    parser.add_argument('ckpt_name', type=str, help='Checkpoint name.')
    parser.add_argument('data_path', type=str, help='Root for the dataset.')

    parser.add_argument('--input_ch_num', type=int, default=3, help='Input channels number.')
    # Optimization options
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    # Checkpoints
    parser.add_argument('--save', type=str, default='./checkpoints/', help='Folder to save checkpoints.')
    parser.add_argument('--log', type=str, default='./logs/', help='Log folder.')
    parser.add_argument('--gpu_num', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers_num', type=int, default=4, help='Pre-fetching threads.')

    parser.add_argument('--image_size', type=int, default=800, help='Train and test image size.')

    args = parser.parse_args()

    model_name = args.model_name
    batch_size = args.batch_size
    model_path = os.path.join(args.save, '{}.pth'.format(args.ckpt_name))

    mean_lst = open('./mean.txt')
    mean = mean_lst.readlines()
    mean = [i.strip() for i in mean]
    mean = [float(i) for i in mean]
    mean_lst.close()

    std_lst = open('./std.txt')
    std = std_lst.readlines()
    std = [i.strip() for i in std]
    std = [float(i) for i in std]
    std_lst.close()

    config = {
        'image_size': [args.image_size, args.image_size],
        'mean': mean,
        'std': std,
        'data_dir': args.data_path,
        'is_trans': True,
        'is_train': False
    }

    test_datasets = SegmentDataset(config)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False,
                             num_workers=args.workers_num, pin_memory=True)

    # Init model, criterion, and optimizer
    model = SmpModel16(model_name, in_channels=args.input_ch_num, out_channels=2)
    # model = UNet(in_channels=args.input_ch_num, out_channels=2)

    loaded_state_dict = torch.load(model_path)
    model_dict = {}
    for key, val in list(loaded_state_dict.items()):
        if 'module' in key:
            # parsing keys for ignoring 'module.' in keys
            model_dict[key[7:]] = val
        else:
            model_dict[key] = val
    model.load_state_dict(model_dict)

    # model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(model, device_ids=list(range(args.gpu_num)))
    model.cuda()

    model.eval()
    print("Test Processing : ")
    with torch.no_grad():
        result = []
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(test_loader)):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            output = model(data)
            # softmax_out = torch.nn.functional.softmax(output, dim=2)
            pred = output.data.max(1)[1]

            result_np = pred.cpu().numpy()
            for i in result_np:
                result.append(i)

    result_path = "./result/{}/".format(args.ckpt_name)
    if os.path.exists(result_path) is not True:
        os.makedirs(result_path)

    count = 1
    for k in result:  # 19
        image_save = np.asarray(k, dtype=np.uint8)
        image_save = Image.fromarray(np.uint8(image_save))
        image_save.save(os.path.join(result_path, "{}.tif".format(count)))
        count += 1
