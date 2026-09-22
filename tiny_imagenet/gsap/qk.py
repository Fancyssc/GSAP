"""Tiny ImageNet QKFormer with the shared, single-width GSAP attention."""
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

if __package__ == 'gsap':  # train.py / test.py launched as scripts
    from baseline.qk import spiking_transformer as QKBackbone
else:
    from ..baseline.qk import spiking_transformer as QKBackbone
from .spiking import GSAPAttention

__all__ = ['qk_gsap']


class QKGSAPTransformer(QKBackbone):
    """Replace all QK/SSA blocks; keep the baseline embeddings and MLPs."""

    def __init__(self, img_size_h=64, img_size_w=64, embed_dims=384,
                 depths=4, backend='cupy', **kwargs):
        if depths < 3:
            raise ValueError('depths must be >= 3 for three QK stages')
        if embed_dims < 32 or embed_dims % 32:
            raise ValueError('embed_dims must be a positive multiple of 32')
        if min(img_size_h, img_size_w) < 4 or img_size_h % 4 or img_size_w % 4:
            raise ValueError('Image dimensions must be positive multiples of 4')
        super().__init__(img_size_h=img_size_h, img_size_w=img_size_w,
                         embed_dims=embed_dims, depths=depths, backend=backend,
                         **kwargs)
        for stage, attr, width, stride in (
            (self.stage1, 'tssa', embed_dims // 4, 1),
            (self.stage2, 'tssa', embed_dims // 2, 2),
            (self.stage3, 'ssa', embed_dims, 4),
        ):
            feature_size = max(img_size_h, img_size_w) // stride
            for block in stage:
                setattr(block, attr, GSAPAttention(
                    width, feature_size=feature_size, backend=backend))


@register_model
def qk_gsap(pretrained=False, cache_dir=None, **kwargs):
    if pretrained:
        raise ValueError('No pretrained Tiny ImageNet weights are provided')
    kwargs.pop('drop_block_rate', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    model = QKGSAPTransformer(**kwargs)
    model.default_cfg = _cfg(input_size=(3, 64, 64),
                             num_classes=model.num_classes, crop_pct=1.0)
    return model
