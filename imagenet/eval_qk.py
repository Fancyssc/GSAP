import argparse
import json
import numpy as np
import os
import socket
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

import util.misc as misc
from util.datasets import build_dataset
from engine_finetune import evaluate
from residual import qk

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluate QKFormer on ImageNet', add_help=False)

    parser.add_argument('--batch_size', default=24, type=int,
                        help='Batch size per GPU for evaluation')
    parser.add_argument('--model', default='QKFormer_10_384', type=str, metavar='MODEL',
                        help='Name of model to evaluate')
    parser.add_argument('--ckpt', '--resume', required=True, type=str, dest='ckpt',
                        help='checkpoint path to evaluate')
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--time_step', default=4, type=int,
                        help='model time step')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of classes')
    parser.add_argument('--device', default='cuda',
                        help='device to use for evaluation')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--result_file', default='', type=str,
                        help='optional json file to save evaluation metrics')

    parser.add_argument('--num_gpus', default=0, type=int,
                        help='number of GPUs for direct launch; 0 uses all visible GPUs')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='enable distributed evaluation')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', '--local_rank', default=-1, type=int, dest='local_rank')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed evaluation')

    return parser


def _is_distributed_launch():
    return (
        ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
        or 'SLURM_PROCID' in os.environ
        or 'OMPI_COMM_WORLD_RANK' in os.environ
    )


def _prepare_eval_resume_args(args):
    args.resume = args.ckpt
    args.eval = True
    return args


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def _distributed_worker(local_rank, args):
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    main(args)


def launch(args):
    if _is_distributed_launch() or args.device != 'cuda':
        main(args)
        return

    if not torch.cuda.is_available():
        main(args)
        return

    num_visible_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus > 0 else num_visible_gpus
    num_gpus = min(num_gpus, num_visible_gpus)

    if num_gpus <= 1:
        main(args)
        return

    args.world_size = num_gpus
    args.dist_eval = True
    if args.dist_url == 'env://':
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(_find_free_port()))

    print(f'Launching distributed evaluation on {num_gpus} GPUs')
    mp.spawn(_distributed_worker, args=(args,), nprocs=num_gpus, join=True)


def main(args):
    misc.init_distributed_mode(args)
    _prepare_eval_resume_args(args)

    if args.distributed:
        args.dist_eval = True
        device = torch.device(args.device, args.gpu)
    else:
        device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_val = build_dataset(is_train=False, args=args)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: eval dataset size is not divisible by process count; duplicate samples may be added.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = qk.__dict__[args.model](T=args.time_step)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None)

    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")

    if args.result_file and misc.is_main_process():
        result_path = Path(args.result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, mode='w', encoding='utf-8') as f:
            json.dump(test_stats, f, indent=2)

    if args.distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('QKFormer ImageNet evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    launch(args)
