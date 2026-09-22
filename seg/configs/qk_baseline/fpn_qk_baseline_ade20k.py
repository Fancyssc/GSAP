_base_ = [
    '../_base_/models/fpn_snn_r50.py',
    # '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_320k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
# crop_size = (32, 32)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        init_cfg=None,
        type='qk_baseline',
        img_size_h=512,
        img_size_w=512,
        patch_size=16,
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=150,
        qkv_bias=False,
        depths=8,
        drop_path_rate=0.1,
        sr_ratios=1,
        T=4,
        decode_mode='snn',
        ),
    neck=dict(
        in_channels=[32, 64, 128, 360],
        out_channels=128,
        act_cfg=None),
    decode_head=dict(
        in_channels=[128, 128, 128, 128],
        channels=128,
        num_classes=150,
        act_cfg=None))

# load_from = checkpoint_file
# resume = checkpoint_file
gpu_multiples = 1  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer

# ===== schedule length (from-scratch) =====
# Training from scratch (no ImageNet ckpt): switch between 160k / 320k here.
# warmup scales with total iters (~2% of training) to stabilize random init.
max_iters = 320000        # set to 320000 for the 320k run
warmup_iters = 6000         # set to 6000 for the 320k run

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999),  weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys={
            'neck': dict(lr_mult=2.0),
            'head': dict(lr_mult=2.0)}
        ),
    # from-scratch stability: clip gradients to avoid early divergence
    clip_grad=dict(max_norm=1.0, norm_type=2)
    )
#
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_iters,
        end=max_iters,
        by_epoch=False,
    )
]
# policy='poly', power=0.9, min_lr=0.0, by_epoch=False
optimizer_config = dict()
# learning policy
lr_config = dict(warmup_iters=warmup_iters)
# runtime settings

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2500)
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
