"""Local timm registrations; use in a process without the original registrations."""
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from ..encoding import Encoder
from .qk import spiking_transformer
from .spiking_gsap import vit_snn as GSAPTransformer
from .spiking_baseline import vit_snn as BaselineTransformer

MODEL_NAMES = ('qk_gsap', 'spiking_gsap', 'spiking_baseline')
__all__ = ['MODEL_NAMES', *MODEL_NAMES]


def _build(cls, pretrained=False, encode_type='direct', backend='cupy',
           input_mean=None, input_std=None, cache_dir=None,
           pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    if pretrained:
        raise ValueError('No pretrained encoder weights are supplied; use a checkpoint')
    if backend not in ('torch', 'cupy'):
        raise ValueError('backend must be torch or cupy')
    drop_block = kwargs.pop('drop_block_rate', None)
    if drop_block not in (None, 0):
        raise ValueError('DropBlock is not implemented')
    defaults = dict(img_size_h=32, img_size_w=32, patch_size=4, in_channels=3,
                    num_classes=10, embed_dims=384, num_heads=8, mlp_ratios=4,
                    depths=4, sr_ratios=1, T=4)
    defaults.update(kwargs)
    if defaults['embed_dims'] < 8 or defaults['embed_dims'] % 8:
        raise ValueError('embed_dims must be >= 8 and divisible by 8')
    if defaults['depths'] < (3 if cls is spiking_transformer else 1):
        raise ValueError('qk_gsap requires depths >= 3; spiking models require depths >= 1')
    if any(defaults[k] < 4 or defaults[k] % 4 for k in ('img_size_h', 'img_size_w')):
        raise ValueError('image dimensions must be positive multiples of 4')
    encoder = Encoder(defaults['T'], encode_type, input_mean, input_std)
    # Construct with Torch so CPU smoke checks never require CuPy to be installed.
    if cls is spiking_transformer:
        defaults['lif_backend'] = 'torch'
    model = cls(**defaults)
    for module in model.modules():
        if isinstance(module, MultiStepLIFNode):
            module.backend = backend
    model.encoder = encoder
    model.default_cfg = _cfg(input_size=(defaults['in_channels'], defaults['img_size_h'],
                                       defaults['img_size_w']),
                             num_classes=defaults['num_classes'], crop_pct=1.0,
                             mean=tuple(input_mean) if input_mean is not None else (0., 0., 0.),
                             std=tuple(input_std) if input_std is not None else (1., 1., 1.))
    return model


@register_model
def qk_gsap(pretrained=False, **kwargs):
    return _build(spiking_transformer, pretrained=pretrained, **kwargs)


@register_model
def spiking_gsap(pretrained=False, **kwargs):
    return _build(GSAPTransformer, pretrained=pretrained, **kwargs)


@register_model
def spiking_baseline(pretrained=False, **kwargs):
    return _build(BaselineTransformer, pretrained=pretrained, **kwargs)
