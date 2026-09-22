"""BrainCog TIM UCF loading recipe with a 128x128 model-input adapter.

Reference: BrainCog-X/Brain-Cog examples/TIM/utils/datasets.py,
get_UCF101DVS_data. Raw decoding and train/test selection are delegated to the
installed BrainCog UCF101DVS class; no guessed parser or new split is used.
"""
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms


class FrameInput:
    """Adapt event frames to the existing DVS models without altering time bins."""
    def __call__(self, frames):
        x = torch.as_tensor(frames, dtype=torch.float32)
        if x.ndim != 4 or x.shape[1] != 2 or min(x.shape) < 1:
            raise ValueError(f'Expected BrainCog frames [T, 2, H, W], got {tuple(x.shape)}')
        if x.shape[-2:] != (128, 128):
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        return x


def build_datasets(root, frames_number=10):
    if frames_number < 1:
        raise ValueError('frames_number must be positive')
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'BrainCog UCF101DVS data directory does not exist: {root}')
    try:
        from braincog.datasets.ucf101_dvs import UCF101DVS
        import tonic
        from tonic import DiskCachedDataset
    except ImportError as error:
        raise ImportError(
            'Use the BrainCog/Tonic environment that already loads your UCF101-DVS data. '
            'It must provide braincog.datasets.ucf101_dvs.UCF101DVS and tonic.DiskCachedDataset. '
            'No alternate dataset parser is selected automatically.'
        ) from error
    result = []
    for train in (True, False):
        # This is the same dataset class and equal-time ToFrame call as TIM.
        raw = UCF101DVS(str(root), train=train, transform=transforms.Compose([
            tonic.transforms.ToFrame(sensor_size=UCF101DVS.sensor_size,
                                     n_time_bins=frames_number),
        ]))
        if len(raw) == 0:
            raise ValueError(f'BrainCog returned an empty {"train" if train else "test"} split at {root}')
        post = [FrameInput()]
        if train:
            post.append(transforms.RandomHorizontalFlip())
        # Reuse the existing Tonic frame caches across all model experiments.
        cache = root / f'{"train" if train else "test"}_cache_{frames_number}'
        cached = DiskCachedDataset(raw, cache_path=str(cache),
                                   transform=transforms.Compose(post))
        result.append(cached)
    return tuple(result)
