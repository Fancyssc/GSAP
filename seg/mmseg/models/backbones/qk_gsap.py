"""Reuse baseline segmentation; replace only its eight attention modules."""
from mmseg.registry import MODELS
from .qk_baseline import qk_baseline
from .spiking_gsap import GSAPAttention

__all__ = ['qk_gsap']


@MODELS.register_module()
def qk_gsap(pretrained=None, **kwargs):
    # Fixed kernels support whole images; range does not grow with the image.
    feature_size = (max(kwargs.get('img_size_h', 224),
                        kwargs.get('img_size_w', 224)) + 15) // 16
    model = qk_baseline(pretrained=pretrained, **kwargs)
    for block in [*model.block3, *model.block4]:
        name = 'tssa' if hasattr(block, 'tssa') else 'ssa'
        dim = block.mlp.fc1_conv.in_channels
        setattr(block, name, GSAPAttention(dim, feature_size))
    return model
