# Inherit every experiment setting; replace only backbone attention.
_base_ = ['../qk_baseline/fpn_qk_baseline_voc12aug.py']

model = dict(backbone=dict(type='qk_gsap'))
