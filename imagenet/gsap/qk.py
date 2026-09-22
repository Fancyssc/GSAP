"""ImageNet QKFormer with GSAP attention and the unchanged QK baseline trunk."""
from timm.models.registry import register_model

if __package__ == 'gsap':  # train_qk.py launched as a script
    from baseline.qk import hierarchical_spiking_transformer as QKBackbone
else:
    from ..baseline.qk import hierarchical_spiking_transformer as QKBackbone
from .spiking import GSAPAttention

__all__ = ['qk_gsap', 'qk_gsap_8_384', 'qk_gsap_8_512', 'qk_gsap_8_768']


class QKGSAPTransformer(QKBackbone):
    """Inherit embeddings, MLPs, residuals, pooling and classifier verbatim."""

    def __init__(self, img_size_h=224, img_size_w=224, embed_dims=512,
                 num_heads=8, depths=10, **kwargs):
        if depths < 4:
            raise ValueError('depths must be >= 4 for three QK stages')
        if min(img_size_h, img_size_w) < 16 or img_size_h % 16 or img_size_w % 16:
            raise ValueError('Image dimensions must be positive multiples of 16')
        super().__init__(img_size_h=img_size_h, img_size_w=img_size_w,
                         embed_dims=embed_dims, num_heads=num_heads,
                         depths=depths, **kwargs)
        for stage, attr, width, stride in (
            (self.stage1, 'tssa', embed_dims // 4, 4),
            (self.stage2, 'tssa', embed_dims // 2, 8),
            (self.stage3, 'attn', embed_dims, 16),
        ):
            for block in stage:
                setattr(block, attr, GSAPAttention(
                    width, feature_size=max(img_size_h, img_size_w) // stride))


@register_model
def qk_gsap(pretrained=False, **kwargs):
    if pretrained:
        raise ValueError('No pretrained ImageNet qk_gsap weights are provided')
    for key in ('pretrained_cfg', 'pretrained_cfg_overlay', 'cache_dir', 'drop_block_rate'):
        kwargs.pop(key, None)
    return QKGSAPTransformer(**kwargs)


@register_model
def qk_gsap_8_384(pretrained=False, **kwargs):
    return qk_gsap(pretrained=pretrained, depths=8, embed_dims=384, **kwargs)


@register_model
def qk_gsap_8_512(pretrained=False, **kwargs):
    return qk_gsap(pretrained=pretrained, depths=8, embed_dims=512, **kwargs)


@register_model
def qk_gsap_8_768(pretrained=False, **kwargs):
    return qk_gsap(pretrained=pretrained, depths=8, embed_dims=768, **kwargs)
