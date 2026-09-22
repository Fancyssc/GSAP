# Inherit every experiment setting; replace only backbone attention.
_base_ = ['../qk_baseline/fpn_qk_baseline_ade20k.py']

model = dict(backbone=dict(type='qk_gsap'))
