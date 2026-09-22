# Align only the model with ADE20K; keep the original VOC training settings.
_base_ = [
    '../_base_/models/fpn_snn_r50.py',
    '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# The backbone has no classification head; VOC output is set in decode_head.
model = {'data_preprocessor': {'size': (512, 512)},
 'type': 'EncoderDecoder',
 'backbone': {'init_cfg': None,
              'type': 'spiking_baseline',
              'img_size_h': 512,
              'img_size_w': 512,
              'patch_size': 16,
              'embed_dims': 360,
              'stage_channels': [32, 64, 128, 360],
              'num_heads': 8,
              'mlp_ratios': 4,
              'in_channels': 3,
              'num_classes': 150,
              'qkv_bias': False,
              'depths': 8,
              'drop_path_rate': 0.1,
              'sr_ratios': 1,
              'T': 4,
              'decode_mode': 'snn'},
 'neck': {'in_channels': [32, 64, 128, 360], 'out_channels': 128, 'act_cfg': None},
 'decode_head': {'in_channels': [128, 128, 128, 128],
                 'channels': 128,
                 'num_classes': 21,
                 'act_cfg': None}}

gpu_multiples = 1

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.005),
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0,
         begin=1500, end=160000, by_epoch=False),
]

optimizer_config = dict()
lr_config = dict(warmup_iters=1500)

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
train_dataloader = dict(batch_size=5)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
