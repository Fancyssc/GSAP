_base_ = [
    '../_base_/models/fpn_snn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_320k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=None,
        type='spiking_baseline',
        img_size_h=512,
        img_size_w=512,
        patch_size=16,
        embed_dims=360,
        stage_channels=[32, 64, 128, 360],
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

gpu_multiples = 1

max_iters = 320000
warmup_iters = 6000

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys={
            'neck': dict(lr_mult=2.0),
            'head': dict(lr_mult=2.0)}
        ),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=warmup_iters),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_iters,
        end=max_iters,
        by_epoch=False,
    )
]

optimizer_config = dict()
lr_config = dict(warmup_iters=warmup_iters)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2500)
train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
