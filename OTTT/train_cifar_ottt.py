from models.cifar_registry import MODEL_FACTORIES
import datetime
import csv
import os
import time
import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import sys
from torch.cuda import amp
from modules import neuron, surrogate
import argparse
import math
from utils import Bar, AverageMeter, accuracy
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchtoolbox.transform import Cutout

_seed_ = 2022
import random
random.seed(_seed_)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)


from models import (
    qkformer_bn_cifar,
    spikformer_bn_cifar,
    spikformer_diff_cifar,
    spiking_vgg,
)


def main():

    parser = argparse.ArgumentParser(description='Classify CIFAR')
    parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=2., type=float)
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', default='./data/CIFAR10', type=str)
    parser.add_argument('-dataset', default='cifar10', type=str)
    parser.add_argument('-out_dir',  default='./output/OTTT/', type=str, help='root dir for saving logs and checkpoint')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')

    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-model', type=str, default='online_spiking_vgg11_ws')
    parser.add_argument('-drop_rate', type=float, default=0.0)
    parser.add_argument('-stochdepth_rate', type=float, default=0.0)
    parser.add_argument('-weight_decay', type=float, default=0.0)
    parser.add_argument('-cnf', type=str)
    parser.add_argument('-T_train', default=None, type=int)
    parser.add_argument('-dts_cache', type=str, default='./dts_cache')
    parser.add_argument('-loss_lambda', type=float, default=0.05)
    parser.add_argument('-online_update', action='store_true')
    parser.add_argument('-aux_weight', default=0.1, type=float,
                        help='exp1: auxiliary classification loss weight')
    parser.add_argument('-route_blend', default=0.25, type=float,
                        help='exp2/3/4: surrogate-route gradient blend in [0, 1]')
    parser.add_argument('-route_decay', default=0.5, type=float,
                        help='exp2/3: causal route EMA decay in [0, 1)')
    parser.add_argument('-proxy_temperature', default=1.0, type=float,
                        help='exp2: positive membrane-proxy temperature')

    parser.add_argument('-gpu-id', default='0', type=str, help='gpu id')

    args = parser.parse_args()
    #print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_data_loader = data.DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=args.j)

    testset = dataloader(root=args.data_dir, train=False, download=False, transform=transform_test)
    test_data_loader = data.DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=args.j)

    # Keep factories here rather than constructing transformer models eagerly:
    # model construction requires the single-step neuron passed below.
    model_dict = MODEL_FACTORIES
    if args.model not in model_dict:
        raise ValueError(
            f"unknown -model {args.model!r}; available: "
            + ', '.join(sorted(model_dict))
        )

    experiment_kwargs = {}
    if args.model == 'spikformer_exp1':
        experiment_kwargs = dict(aux_weight=args.aux_weight)
    elif args.model == 'spikformer_exp2':
        experiment_kwargs = dict(route_blend=args.route_blend,
                                 route_decay=args.route_decay,
                                 proxy_temperature=args.proxy_temperature)
    elif args.model == 'spikformer_exp3':
        experiment_kwargs = dict(route_blend=args.route_blend,
                                 route_decay=args.route_decay)
    elif args.model == 'spikformer_exp4':
        experiment_kwargs = dict(route_blend=args.route_blend)

    # with grad_with_rate = True, require_wrap does not need to be set explicitly
    net = model_dict[args.model](single_step_neuron=neuron.OnlineLIFNode, tau=args.tau, surrogate_function=surrogate.Sigmoid(alpha=4.),
                                 track_rate=True, c_in=3, num_classes=num_classes, neuron_dropout=args.drop_rate, grad_with_rate=True, fc_hw=1, v_reset=None,
                                 **experiment_kwargs)
    #print(net)
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()


    optimizer = None
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    ottt_mode = 'ottt_o' if args.online_update else 'ottt_a'
    out_dir = os.path.join(args.out_dir, f'{args.model}_{ottt_mode}_T_{args.T}_T_train_{args.T_train}_')
    for name, value in experiment_kwargs.items():
        out_dir += f'{name}_{value}_'
    if args.lr_scheduler == 'CosALR':
        out_dir += f'CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)
        #assert args.resume is not None

    accuracy_file = os.path.join(out_dir, 'epoch_acc.csv')
    accuracy_file_exists = os.path.exists(accuracy_file)
    accuracy_mode = 'a' if args.resume and accuracy_file_exists else 'w'
    with open(accuracy_file, accuracy_mode, newline='', encoding='utf-8') as f:
        if accuracy_mode == 'w':
            csv.writer(f).writerow(['epoch', 'train_acc', 'test_acc', 'max_test_acc'])
    
    criterion_mse = nn.MSELoss(reduce=True)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(train_data_loader))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0
        for frame, label in train_data_loader:
            batch_idx += 1
            frame = frame.float().cuda()
            t_step = args.T

            label = label.cuda()

            batch_loss = 0
            if not args.online_update:
                optimizer.zero_grad()
            for t in range(t_step):
                if args.online_update:
                    optimizer.zero_grad()
                input_frame = frame
                if args.amp:
                    with amp.autocast():
                        if t == 0:
                            out_fr = net(input_frame, init=True)
                            total_fr = out_fr.clone().detach()
                        else:
                            out_fr = net(input_frame)
                            total_fr += out_fr.clone().detach()
                            #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                        if args.loss_lambda > 0.0:
                            label_one_hot = F.one_hot(label, num_classes).float()
                            mse_loss = criterion_mse(out_fr, label_one_hot)
                            loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                        else:
                            loss = F.cross_entropy(out_fr, label) / t_step
                        if args.model == 'spikformer_exp1':
                            loss = loss + net.auxiliary_loss(label) / t_step
                    scaler.scale(loss).backward()

                    if args.online_update:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    if t == 0:
                        out_fr = net(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, num_classes).float()
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                    if args.model == 'spikformer_exp1':
                        loss = loss + net.auxiliary_loss(label) / t_step
                    loss.backward()

                    if args.online_update:
                        optimizer.step()

                batch_loss += loss.item()
                train_loss += loss.item() * label.numel()
            if not args.online_update:
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
            losses.update(batch_loss, input_frame.size(0))
            top1.update(prec1.item(), input_frame.size(0))
            top5.update(prec5.item(), input_frame.size(0))


            train_samples += label.numel()
            train_acc += (total_fr.argmax(1) == label).float().sum().item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx,
                        size=len(train_data_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(test_data_loader))

        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                batch_idx += 1
                frame = frame.float().cuda()
                label = label.cuda()
                t_step = args.T
                total_loss = 0

                for t in range(t_step):
                    input_frame = frame
                    if t == 0:
                        out_fr = net(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                        #total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                    if args.loss_lambda > 0.0:
                        label_one_hot = F.one_hot(label, num_classes).float()
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step
                    total_loss += loss

                test_samples += label.numel()
                test_loss += total_loss.item() * label.numel()
                test_acc += (total_fr.argmax(1) == label).float().sum().item()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
                losses.update(total_loss, input_frame.size(0))
                top1.update(prec1.item(), input_frame.size(0))
                top5.update(prec5.item(), input_frame.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=len(test_data_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            )
                bar.next()
        bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc,
            'args': vars(args)
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        with open(accuracy_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, train_acc, test_acc, max_test_acc])
        #for item in sys.argv:
        #    print(item, end=' ')
        #print('')
        #print(args)
        #print(out_dir)
        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
