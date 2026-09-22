# Inherit every experiment setting; replace only backbone attention.
_base_ = ['../spiking_baseline/fpn_spiking_baseline_ade20k.py']

model = dict(backbone=dict(type='spiking_gsap'))
