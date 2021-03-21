import sys

sys.path.append('../../')

import argparse
import os

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import json
import tqdm
import random
from tools.classifi_mode.classifi_datasets import ClassifiDataset, get_model

# os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on IEEE GRSS')
    # Positional arguments
    parser.add_argument('model_name', type=str, help='Model name.')
    parser.add_argument('data_path', type=str, help='Root for the dataset.')
    parser.add_argument('input_ch_num', type=int, help='Input channels number.')
    parser.add_argument('batch_size', type=int, help='Batch size.')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='The Learning Rate.')
    # parser.add_argument('--focal_loss', type=bool, default=False, help='If use focal loss.')
    # Checkpoints
    parser.add_argument('--save', type=str, default='./checkpoints/', help='Folder to save checkpoints.')
    parser.add_argument('--log', type=str, default='./logs/', help='Log folder.')
    parser.add_argument('--gpu_num', type=int, default=2, help='0 = CPU.')
    parser.add_argument('--workers_num', type=int, default=8, help='Pre-fetching threads.')

    parser.add_argument('--image_size', type=int, default=64, help='Train and test image size.')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizers choice.')

    args = parser.parse_args()

    state = {
        # 'model_name': args.model_name,
        'epoch': 0,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'train_loss': 0.0,
        'test_acc': 0.0,
        'best_acc': 0.0,
        'image_size': args.image_size,
        'best_loss': 0.0,
    }

    # if args.focal_loss:
    #     state['focal_loss'] = True

    # Init checkpoints
    ckpt_save_path = os.path.join(args.save, args.model_name)
    if not os.path.isdir(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    acc_ckpt = os.path.join(ckpt_save_path, 'acc.pth')
    loss_ckpt = os.path.join(ckpt_save_path, 'loss.pth')

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, '{}.txt'.format(args.model_name)), 'w')
    log.write(json.dumps(state) + '\n')

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
        'txt_dir': "/root/data/cls/classifi_data/train.txt",
        'is_trans': True
    }

    # model = get_model(args.model_name, args.input_ch_num, 2, 64)
    # print(model)

    datasets = ClassifiDataset(config)

    train_num = int(0.9 * len(datasets))
    val_num = len(datasets) - train_num
    split_num = random.randint(0, 100)
    train_datasets, val_datasets = random_split(datasets, [train_num, val_num],
                                                generator=torch.Generator().manual_seed(split_num))

    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers_num, pin_memory=True)
    val_loader = DataLoader(val_datasets, batch_size=16, shuffle=False,
                            num_workers=args.workers_num, pin_memory=True)

    # Init model, criterion, and optimizer
    model = get_model(args.model_name, args.input_ch_num, 2, args.image_size)
    print(model)

    model = torch.nn.DataParallel(model, device_ids=list(range(args.gpu_num)))
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), state['lr'], momentum=0.8, nesterov=True)
    else:
        optimizer = torch.optim.Adamax(model.parameters(), state['lr'])

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=85, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, args.lr * 0.01)

    # if args.focal_loss:
    #     loss_func = FocalLossV3(alpha=0.75)
    #     loss_func.cuda()
    # else:
    loss_func = torch.nn.CrossEntropyLoss()

    # Main loop
    best_accuracy = 0.0
    best_loss = 100.0
    for epoch in range(args.epochs):
        state['epoch'] = epoch
        print("epoch {}-start : lr:{}".format(epoch, state['lr']))

        # start train
        model.train()
        loss_avg = 0.0

        for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())
            optimizer.zero_grad()

            with autocast():
                output = model(data)
                loss = loss_func(output, target)

            loss.backward()
            optimizer.step()
            loss_avg = loss_avg * 0.1 + float(loss) * 0.9

        state['train_loss'] = loss_avg
        print("train_loss : {}".format(state['train_loss']))

        model.eval()
        loss_avg = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            output = model(data)
            loss = loss_func(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # test loss average
            loss_avg += float(loss)

        test_loss = loss_avg / len(val_loader)
        state['test_acc'] = correct / len(val_loader.dataset)
        print("test_loss : {} test_accuracy : {}".format(test_loss, state['test_acc']))

        scheduler.step()
        state['lr'] = scheduler.get_last_lr()

        # save ckpt
        if state['test_acc'] > best_accuracy:
            best_accuracy = state['test_acc']
            state['best_acc'] = best_accuracy
            torch.save(model.state_dict(), acc_ckpt)

        if state['train_loss'] < best_loss:
            best_loss = state['train_loss']
            state['best_loss'] = best_loss
            torch.save(model.state_dict(), loss_ckpt)

        if (epoch + 1) % 10 == 0:
            epoch_ckpt = os.path.join(ckpt_save_path, '{}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), epoch_ckpt)

        log.write('%s\n' % json.dumps(state))
        log.flush()
        print("Best accuracy: {} Best loss: {}".format(best_accuracy, best_loss))

    log.close()
