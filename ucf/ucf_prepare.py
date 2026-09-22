"""Populate the BrainCog/Tonic UCF event-frame cache before training."""
import argparse
from dataset import build_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--T', '--frames-number', dest='T', type=int, default=10)
    args = parser.parse_args()
    for name, dataset in zip(('train', 'test'), build_datasets(args.data_path, args.T)):
        for i in range(len(dataset)):
            frames, label = dataset[i]
            if i % 100 == 0:
                print(f'{name}: {i + 1}/{len(dataset)}, shape={tuple(frames.shape)}', flush=True)


if __name__ == '__main__':
    main()
