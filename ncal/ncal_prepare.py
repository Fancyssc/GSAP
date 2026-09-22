"""Prepare event/frame caches from manually downloaded N-Caltech101 files."""
import argparse
from dataset import NCaltech101


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--frames-number', type=int, default=16)
    args = parser.parse_args()
    dataset = NCaltech101(root=args.data_path, data_type='frame',
                         frames_number=args.frames_number, split_by='number')
    print(f'Prepared {len(dataset)} samples, {len(dataset.classes)} classes.')


if __name__ == '__main__':
    main()
