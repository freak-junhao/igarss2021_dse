import sys

sys.path.append('../../')

import argparse
import os

import torch
from torch.utils.data import DataLoader

import tqdm
from tools.cls_next.classifi_datasets import ClassifiDataset, get_model
from tools.cls_next.classifi_data_process import trans_to_image

# os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'


def get_args():
    parser = argparse.ArgumentParser(description='Trains ResNeXt on IEEE GRSS')
    # Positional arguments
    parser.add_argument('model_name', type=str, help='Model name.')
    parser.add_argument('ckpt_name', type=str, help='Checkpoint name.')
    parser.add_argument('data_path', type=str, help='Root for the dataset.')
    parser.add_argument('input_ch_num', type=int, help='Input channels number.')
    # Optimization options
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    # Checkpoints
    parser.add_argument('--save', type=str, default='./checkpoints/', help='Folder to save checkpoints.')
    parser.add_argument('--log', type=str, default='./logs/', help='Log folder.')
    parser.add_argument('--gpu_num', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers_num', type=int, default=8, help='Pre-fetching threads.')

    parser.add_argument('--image_size', type=int, default=64, help='Train and test image size.')

    args = parser.parse_args()
    return args


def test_main():
    args = get_args()  # divide args part and call it as function

    model_name = args.model_name
    batch_size = args.batch_size
    model_path = os.path.join(args.save, '{}.pth'.format(args.ckpt_name))

    # prepare test data parts
    mean_lst = open('./txt_files/mean.txt')
    mean = mean_lst.readlines()
    mean = [i.strip() for i in mean]
    mean = [float(i) for i in mean]
    mean_lst.close()

    std_lst = open('./txt_files/std.txt')
    std = std_lst.readlines()
    std = [i.strip() for i in std]
    std = [float(i) for i in std]
    std_lst.close()

    config = {
        'image_size': [args.image_size, args.image_size],
        'mean': mean,
        'std': std,
        'data_dir': args.data_path,
        'txt_dir': "/root/data/cls/classifi_data/val.txt",
        'is_trans': True
    }

    test_datasets = ClassifiDataset(config)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False,
                             num_workers=args.workers_num, pin_memory=True)

    # initialize model and load from checkpoint
    model = get_model(model_name, args.input_ch_num, 2, args.image_size)

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
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
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
    trans_to_image(result, args.data_path, result_path)


if __name__ == '__main__':
    test_main()
