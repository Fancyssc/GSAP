"""Static image encoders following BrainCog's weighted TTFS/phase semantics.

Reference: https://github.com/BrainCog-X/Brain-Cog/blob/main/braincog/base/encoder/encoder.py
Input: floating BCHW. Output: contiguous TBCHW on the input device.
Rate returns FP32 exactly as BrainCog; other encodings preserve input dtype.
"""
import math

import torch
from torch import nn

ENCODINGS = ('direct', 'ttfs', 'rate', 'phase')


class Encoder(nn.Module):
    def __init__(self, step=4, encode_type='direct', input_mean=None, input_std=None):
        super().__init__()
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError('step must be a positive integer')
        if encode_type not in ENCODINGS:
            raise ValueError(f'encode_type must be one of {ENCODINGS}')
        if (input_mean is None) != (input_std is None):
            raise ValueError('input_mean and input_std must be supplied together')
        self.step = step
        self.encode_type = encode_type
        if input_mean is not None:
            if not input_mean or len(input_mean) != len(input_std):
                raise ValueError('mean/std must have the same nonzero channel count')
            if not all(math.isfinite(v) for v in (*input_mean, *input_std)):
                raise ValueError('mean/std must be finite')
            if any(v <= 0 for v in input_std):
                raise ValueError('std must be positive')
        # Nonpersistent buffers preserve compatibility with direct checkpoints.
        self.register_buffer('input_mean', None if input_mean is None else
                             torch.tensor(input_mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer('input_std', None if input_std is None else
                             torch.tensor(input_std).view(1, -1, 1, 1), persistent=False)

    def forward(self, inputs, generator=None):
        if inputs.ndim != 4 or not inputs.is_floating_point():
            raise ValueError('Encoder expects floating BCHW static images')
        if self.encode_type == 'direct':
            return inputs.unsqueeze(0).repeat(self.step, 1, 1, 1, 1)
        with torch.no_grad():
            if self.encode_type == 'rate':
                # Match BrainCog on the exact supplied input: no preprocessing,
                # default RNG dtype, strict comparison, and FP32 spike output.
                shape = (self.step,) + inputs.shape
                return (inputs > torch.rand(shape, device=inputs.device,
                                            generator=generator)).float()
            # Calculate quantization/probabilities in FP32 for FP16/BF16 inputs.
            x = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
            if self.input_mean is not None:
                if inputs.shape[1] != self.input_mean.shape[1]:
                    raise ValueError('mean/std channel count does not match input')
                x = x * self.input_std.to(x) + self.input_mean.to(x)
            # Augmentations can leave the nominal [0, 1] interval.
            x = x.clamp(0, 1)
            if self.encode_type == 'ttfs':
                # Bin ((T-t-1)/T, (T-t)/T], amplitude 1/(t+1); zero is silent.
                times = torch.arange(self.step, device=x.device).view(-1, 1, 1, 1, 1)
                scaled = x.unsqueeze(0) * self.step
                result = ((scaled <= self.step - times) &
                          (scaled > self.step - times - 1)).to(x.dtype) / (times + 1)
            else:
                # Eight-bit, MSB first, weights 1, 1/2, ..., 1/128, repeated.
                # Saturate x=1 to 255: BrainCog's 256 would lose all low 8 bits.
                quantized = (x * 256).long().clamp(0, 255)
                phases = torch.arange(self.step, device=x.device) % 8
                bits = (quantized.unsqueeze(0) >> (7 - phases).view(-1, 1, 1, 1, 1)) & 1
                result = bits * (2.0 ** (-phases.to(x.dtype))).view(-1, 1, 1, 1, 1)
            return result.to(dtype=inputs.dtype).contiguous()

    def extra_repr(self):
        return f'step={self.step}, encode_type={self.encode_type!r}'
