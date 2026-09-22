"""Validation using the same config, model registrations and metrics as training."""
from contextlib import suppress

import torch
from spikingjelly.clock_driven import functional
from timm.data import create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint
from timm.utils import setup_default_logging

from train import _parse_args, validate
from dataset import TinyImageNet


def main():
    setup_default_logging()
    args, _ = _parse_args()
    checkpoint = args.initial_checkpoint or args.resume
    if not checkpoint:
        raise ValueError('Specify --initial-checkpoint (or --resume) for evaluation')
    args.rank = args.local_rank = 0
    args.world_size = 1
    args.distributed = False
    if args.device.isdigit():
        args.device = f'cuda:{args.device}'
    torch.cuda.set_device(args.device)
    args.prefetcher = not args.no_prefetcher
    model = create_model(
        args.model, pretrained=False, T=args.T, backend=args.backend,
        img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=args.patch_size, embed_dims=args.dim, depths=args.depths,
        num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
        num_classes=args.num_classes, in_channels=3, sr_ratios=1,
        drop_rate=args.drop, drop_path_rate=args.drop_path)
    load_checkpoint(model, checkpoint, use_ema=args.model_ema)
    model = model.to(args.device)
    data_config = resolve_data_config(vars(args), model=model)
    dataset = TinyImageNet(args.data_dir, args.val_split)
    if len(dataset.classes) != args.num_classes:
        raise ValueError('Dataset class count does not match num_classes')
    loader = create_loader(
        dataset, input_size=data_config['input_size'], batch_size=args.val_batch_size,
        is_training=False, use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'], mean=data_config['mean'],
        std=data_config['std'], crop_pct=data_config['crop_pct'],
        num_workers=args.workers, pin_memory=args.pin_mem)
    autocast = torch.cuda.amp.autocast if args.amp or args.native_amp else suppress
    functional.reset_net(model)
    print(validate(model, loader, torch.nn.CrossEntropyLoss().to(args.device),
                   args, amp_autocast=autocast))


if __name__ == '__main__':
    main()
