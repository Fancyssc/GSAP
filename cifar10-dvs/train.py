import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model, is_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import SoftTargetCrossEntropy
import autoaugment


from baseline import qkformer  # register qk_baseline and QKFormer
from baseline import sdt  # register sdt_baseline and SDT
from gsap import spiking, qk  # register spiking_gsap and qk_gsap

try:

    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='spiking_gsap', help='model')
    parser.add_argument('--dataset', default='cifar10dvs', help='dataset')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='./data/CIFAR10DVS', help='dataset')
    parser.add_argument('--device', default='cuda', help="device, e.g. '4' / 'cuda:4' / 'cuda' / 'cpu'")
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output/train', help='path where to save')
    parser.add_argument('--experiment', default='', type=str,
                        help='experiment name, also used as the default wandb run name')
    parser.add_argument('--stsa-mode', choices=('full', 'window', 'lowrank'), default='full',
                        help='similarity matrix mode used by STSA')
    parser.add_argument('--stsa-evolve-layers', type=int, default=3,
                        help='number of depthwise evolution layers on the similarity graph (0 disables)')
    parser.add_argument('--stsa-evolve-kernel', type=int, nargs='+', default=[3],
                        help='evolution kernels: one value expands as k, k+2, ... per layer; '
                             'a list (e.g. 3 5 7) is used as-is and overrides stsa-evolve-layers')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        # default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--wandb-project', default='GSAP', type=str,
                        help='wandb project name')
    parser.add_argument('--wandb-name', default='', type=str,
                        help='wandb run name, defaults to experiment')
    parser.add_argument('--wandb-tags', default=None, nargs='*',
                        help='optional wandb tags')
    parser.add_argument('--wandb-mode', default='online', choices=('online', 'offline', 'disabled'),
                        help='wandb run mode')
    parser.add_argument('--T', default=16, type=int, help='number of event frames')
    parser.add_argument('--backend', default='cupy', choices=('cupy', 'torch'),
                        help='LIF backend for GSAP, QK and SDT models')
    parser.add_argument('--simfield-spatial-kernel', type=int, default=3,
                        help='odd spatial neighborhood size for ST-SimField')
    parser.add_argument('--simfield-dilation', type=int, default=1,
                        help='spatial offset dilation for ST-SimField')
    parser.add_argument('--simfield-temporal-lags', type=int, nargs='+', default=[0, 1],
                        help='causal K lags used by ST-SimField')
    parser.add_argument('--simfield-temporal-kernel', type=int, default=3,
                        help='causal relation-mixer kernel along time')
    parser.add_argument('--simfield-similarity', choices=('and', 'xnor'), default='and',
                        help='binary Q/K similarity used by ST-SimField')
    parser.add_argument('--simfield-route-scale', type=float, default=None,
                        help='fixed scale after binary K routing; default is power-of-two')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=96, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=75, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 75)')
    args = parser.parse_args()
    return args


def _safe_path_name(value):
    value = str(value).strip()
    value = value.replace(os.sep, '_').replace('/', '_').replace('\\', '_').replace(' ', '_')
    return value or 'default'


def init_wandb(args, output_dir):
    if not args.log_wandb or not utils.is_main_process():
        return False
    if not has_wandb:
        raise RuntimeError("You've requested wandb logging but the wandb package is not installed. "
                           "Install it with `pip install wandb`.")

    project = args.wandb_project
    name = args.wandb_name or args.experiment or os.path.basename(output_dir)
    try:
        run = wandb.init(
            project=project,
            name=name,
            tags=args.wandb_tags,
            config=vars(args),
            mode=args.wandb_mode)
    except Exception as exc:
        raise RuntimeError("Failed to initialize wandb. Run `wandb login` first, or set "
                           "`WANDB_API_KEY` in this environment.") from exc
    if run is None:
        raise RuntimeError("wandb.init() returned None. Check your wandb login and WANDB_MODE settings.")
    print(f'Initialized wandb run: {getattr(run, "url", "")}')
    return True


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]
        N,T,C,H,W = image.shape

        if aug != None:
            image = torch.stack([(aug(image[i])) for i in range(N)])
            # image = image.reshape(N,T,C,H,W)

        if trival_aug != None:
            # image = image.flatten(0,1).contiguous()
            image = torch.stack([(trival_aug(image[i])) for i in range(N)])
            # image = image.reshape(N,T,C,H,W).contiguous()

        if mixup_fn is not None:
            # timm Mixup/CutMix expects NCHW. Share spatial boxes across all
            # frames and both polarities, then restore the original sequence.
            image, target = mixup_fn(image.flatten(1, 2), target)
            image = image.reshape(N, T, C, H, W)
            target_for_compu_acc = target.argmax(dim=-1)


        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5

def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')
    dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler

def main(args):
    if args.T < 1 or (args.T_train is not None and not 1 <= args.T_train <= args.T):
        raise ValueError('Require T >= 1 and 1 <= T_train <= T when specified')
    if not is_model(args.model):
        raise ValueError(f'Unknown model: {args.model!r}. Available local models: '
                         'spiking_gsap, qk_gsap, qk_baseline, QKFormer, sdt_baseline, SDT')

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.
    best_epoch = None
    final_train_loss = None
    final_train_acc1 = None
    final_train_acc5 = None
    final_test_loss = None
    final_test_acc1 = None
    final_test_acc5 = None


    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.experiment:
        exp_name = args.experiment
    else:
        exp_name = args.model
        if args.model.endswith('stsa'):
            exp_name = f"{exp_name}-T{args.T}-L2-h256"
    exp_name = _safe_path_name('_'.join([exp_name, run_time]))
    output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)
    wandb_active = init_wandb(args, output_dir)

    device_str = str(args.device)
    if device_str.isdigit():
        device_str = f'cuda:{device_str}'
    device = torch.device(device_str)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    data_path = args.data_path

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        # sampler=test_sampler,
        drop_last=False,
        pin_memory=True)
    # print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    model = create_model(
        model_name=args.model,
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0. if args.model in ('spiking_gsap', 'qk_gsap', 'qk_baseline', 'QKFormer', 'sdt_baseline', 'SDT') else 0.1,
        drop_block_rate=None,
        **(dict(T=args.T, num_classes=args.num_classes, backend=args.backend)
           if args.model in ('spiking_gsap', 'qk_gsap', 'qk_baseline', 'QKFormer', 'sdt_baseline', 'SDT') else {}),
        **(dict(
            simfield_spatial_kernel=args.simfield_spatial_kernel,
            simfield_dilation=args.simfield_dilation,
            simfield_temporal_lags=args.simfield_temporal_lags,
            simfield_temporal_kernel=args.simfield_temporal_kernel,
            simfield_similarity=args.simfield_similarity,
            simfield_route_scale=args.simfield_route_scale,
        ) if args.model.endswith('SimField') else {}),
    )
    print("Creating model")
    if args.model == 'spiking_gsap':
        print(f'Attention: {type(model.block[0].attn).__name__}; '
              f'T={args.T}, T_train={args.T_train}, depth={model.depths}, '
              f'dim={model.num_features}, feature_grid='
              f'{model.patch_embed.H}x{model.patch_embed.W}')
    if args.model == 'qk_gsap':
        print(f'Attention: {type(model.stage1[0].tssa).__name__}; '
              f'T={args.T}, T_train={args.T_train}, depth={model.depths}, '
              f'dim={model.num_features}, feature_sizes={model.feature_sizes}')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
    criterion_train = (SoftTargetCrossEntropy() if mixup_active else
                       nn.CrossEntropyLoss(label_smoothing=args.smoothing)).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=1e-5)
    #op = torch.optim.adamw()
    start_epoch = 0
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']
        best_epoch = checkpoint.get('best_epoch')

    if args.test_only:

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        if wandb_active:
            wandb.log({
                'test/loss': test_loss,
                'test/acc1': test_acc1,
                'test/acc5': test_acc5,
            }, step=args.start_epoch)
            wandb.finish()

        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(os.path.join(output_dir, 'logs', 'train'), purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(os.path.join(output_dir, 'logs', 'te'), purge_step=purge_step_te)
        with open(os.path.join(output_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')


    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        if utils.is_main_process():
            print('\n' + '=' * 80)
            print(f'Epoch {epoch + 1}/{num_epochs} (epoch_index={epoch})')
            print('=' * 80)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if mixup_fn is not None and args.mixup_off_epoch > 0 and epoch >= args.mixup_off_epoch:
            mixup_fn.mixup_enabled = False
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn)
        if train_tb_writer is not None and utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step(epoch + 1)

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        final_train_loss = train_loss
        final_train_acc1 = train_acc1
        final_train_acc5 = train_acc5
        final_test_loss = test_loss
        final_test_acc1 = test_acc1
        final_test_acc5 = test_acc5
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            best_epoch = epoch
            save_max = True

        if wandb_active:
            wandb.log({
                'train/loss': train_loss,
                'train/acc1': train_acc1,
                'train/acc5': train_acc5,
                'test/loss': test_loss,
                'test/acc1': test_acc1,
                'test/acc5': test_acc5,
                'best/test_acc1': max_test_acc1,
                'best/test_acc5_at_best_acc1': test_acc5_at_max_test_acc1,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            }, step=epoch)


        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                'best_epoch': best_epoch,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        if utils.is_main_process():
            print('-' * 80)
            print(
                f'Epoch {epoch + 1}/{num_epochs} summary | '
                f'best_test_acc1={max_test_acc1:.4f}, test_acc5_at_best={test_acc5_at_max_test_acc1:.4f}, \n'
                f'train_loss={train_loss:.6f}, train_acc1={train_acc1:.4f}, train_acc5={train_acc5:.4f} \n| '
                f'test_loss={test_loss:.6f}, test_acc1={test_acc1:.4f}, test_acc5={test_acc5:.4f} \n| '
                f'best_epoch={best_epoch}, elapsed={total_time_str}'
            )
            print(f'Output dir: {output_dir}')
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.is_main_process():
        print('\n' + '=' * 80)
        print('Experiment summary')
        print('=' * 80)
        print(f'Experiment: {args.experiment or "N/A"}')
        print(f'Model: {args.model}')
        print(f'Epochs: {args.start_epoch} -> {num_epochs - 1}')
        print(f'Best Top-1 Acc: {max_test_acc1:.4f}')
        print(f'Top-5 Acc at Best Top-1: {test_acc5_at_max_test_acc1:.4f}')
        print(f'Best Epoch: {best_epoch}')
        if final_test_acc1 is not None:
            print(
                f'Final Epoch Result: '
                f'train_loss={final_train_loss:.6f}, train_acc1={final_train_acc1:.4f}, train_acc5={final_train_acc5:.4f} | '
                f'test_loss={final_test_loss:.6f}, test_acc1={final_test_acc1:.4f}, test_acc5={final_test_acc5:.4f}'
            )
        print(f'Total Training Time: {total_time_str}')
        print(f'Output dir: {output_dir}')

    if wandb_active:
        wandb.finish()

    return max_test_acc1


if __name__ == "__main__":
    args = parse_args()
    main(args)
