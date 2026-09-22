"""Read official Tiny ImageNet or class-folder validation without moving files."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = Path(root).expanduser()
        self.transform = transform
        self.target_transform = target_transform
        split = 'val' if split == 'validation' else split
        if split not in ('train', 'val'):
            raise ValueError('Use train or val; the official test split has no labels')
        wnids = self.root / 'wnids.txt'
        if wnids.is_file():
            self.classes = sorted(set(wnids.read_text().split()))
        else:
            self.classes = sorted(p.name for p in (self.root / 'train').iterdir() if p.is_dir())
        if not self.classes:
            raise ValueError(f'No classes found in {self.root}')
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = []
        folder = self.root / split
        annotations = folder / 'val_annotations.txt'
        if split == 'val' and annotations.is_file() and (folder / 'images').is_dir():
            for line in annotations.read_text().splitlines():
                if not line.strip():
                    continue
                name, wnid, *_ = line.split()
                image = folder / 'images' / name
                if wnid not in self.class_to_idx:
                    raise ValueError(f'Unknown validation class: {wnid}')
                if not image.is_file():
                    raise FileNotFoundError(image)
                self.samples.append((str(image), self.class_to_idx[wnid]))
        else:
            extensions = {'.jpeg', '.jpg', '.png', '.bmp'}
            for wnid in self.classes:
                images = sorted(p for p in (folder / wnid).rglob('*') if p.suffix.lower() in extensions)
                if not images:
                    raise FileNotFoundError(f'No images for {split}/{wnid} in {self.root}')
                self.samples.extend((str(p), self.class_to_idx[wnid]) for p in images)
        if not self.samples:
            raise ValueError(f'No labeled images in {folder}')
        self.targets = [target for _, target in self.samples]
        self.imgs = self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
