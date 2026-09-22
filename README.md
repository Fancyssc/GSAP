# GSAP

This repository contains the official implementation of **GSAP** (Gated Spiking Axial Propagation attention), a spiking attention module for spiking transformer networks. GSAP replaces the QK self-attention of spiking transformers with single-width axial propagation (row-then-column depthwise convolution), a local branch, and a binary gate, and is evaluated on image classification, neuromorphic event-stream classification, and semantic segmentation.

## Repository layout

| Directory | Task | Entry points |
|---|---|---|
| `cifar10/`, `cifar100/` | CIFAR-10 / CIFAR-100 classification | `train.py`, `test.py` |
| `tiny_imagenet/` | Tiny ImageNet classification | `train.py`, `test.py` |
| `imagenet/` | ImageNet-1k classification | `train.py`, `train_qk.py`, `test.py`, `eval_qk.py` |
| `cifar10-dvs/` | CIFAR10-DVS event classification | `dvsc10_download.py`, `train.py` |
| `ncal/` | N-Caltech101 event classification | `ncal_prepare.py`, `train.py`, `run_ncal.sh` |
| `ucf/` | UCF101-DVS event classification | `ucf_prepare.py`, `train.py` |
| `seg/` | Semantic segmentation (ADE20K, Pascal VOC) with MMSegmentation | `tools/train.py`, `tools/test.py` |
| `ablation/` | GSAP branch ablations on CIFAR | `train.py`, `run_all.sh`, `check_models.py` |
| `encoder/` | Input encoding schemes on CIFAR | `train.py`, `check_models.py` |

In each directory, the GSAP models live in `gsap/spiking.py` (Spikingformer backbone) and `gsap/qk.py` (QKFormer backbone); the unchanged comparison baselines live in `baseline/`.

## Requirements

- Python >= 3.8
- PyTorch (with CUDA for training)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [SpikingJelly](https://github.com/fangwei123/spikingjelly) (`spikingjelly`)
- [tonic](https://github.com/neuromorphs/tonic) and [BrainCog](https://github.com/BrainCog-X/Brain-Cog) datasets (for CIFAR10-DVS / N-Caltech101 / UCF101-DVS)
- For segmentation: [MMEngine](https://github.com/open-mmlab/mmengine) and the bundled `seg/mmseg` package (see `seg/requirements/`)
- Optional: `cupy` for the CUDA LIF backend (`--backend cupy`); use `--backend torch` to run without it

## Data preparation

All scripts expect datasets under `./data/` by default (override with `--data-dir` / `--data-path`):

```
data/
├── cifar10/                  # torchvision CIFAR-10
├── cifar100/                 # torchvision CIFAR-100
├── tiny-imagenet-200/        # http://cs231n.stanford.edu/tiny-imagenet-200.zip
├── imagenet/                 # ILSVRC2012 train/ + val/
├── CIFAR10DVS/               # see below
├── NCaltech101/              # see below
├── UCF101DVS/                # see below
├── ADEChallengeData2016/     # https://groups.csail.mit.edu/vision/datasets/ADE20K/
└── VOCdevkit/VOC2012/        # http://host.robots.ox.ac.uk/pascal/VOC/
```

CIFAR-10/100 are downloaded automatically by torchvision. For the event-stream datasets:

```bash
# CIFAR10-DVS (downloads and extracts the zip parts)
python cifar10-dvs/dvsc10_download.py

# N-Caltech101 (download the dataset manually, then build the frame cache)
python ncal/ncal_prepare.py --data-path ./data/NCaltech101 --frames-number 16

# UCF101-DVS (download the dataset manually, then build the frame cache)
python ucf/ucf_prepare.py --data-path ./data/UCF101DVS --T 10
```

## Training and evaluation

Run every command from the repository root unless noted otherwise.

### CIFAR-10 / CIFAR-100

```bash
# GSAP on the Spikingformer backbone
python cifar10/train.py --model spiking_gsap
python cifar100/train.py --model spiking_gsap

# GSAP on the QKFormer backbone
python cifar10/train.py --model qk_gsap
python cifar100/train.py --model qk_gsap

# Baselines (Spikingformer, QKFormer, SDT, Spikformer, TEFormer, CML)
python cifar10/train.py --model Spikingformer

# Evaluation from a checkpoint
python cifar10/test.py --model spiking_gsap --resume /path/to/checkpoint.pth.tar
```

Hyperparameters come from `cifar10/cifar10.yml` / `cifar100/cifar100.yml` (`-c/--config`); pass `--help` for the full set of overrides (batch size, epochs, time step `--T`, learning rate, wandb logging, etc.).

### Tiny ImageNet

```bash
# --model in {spiking_baseline, spiking_gsap, qk_baseline, qk_gsap, sdt_baseline, spikf_baseline}
python tiny_imagenet/train.py --model spiking_gsap
python tiny_imagenet/test.py  --model spiking_gsap --resume /path/to/checkpoint.pth.tar
```

### ImageNet

```bash
# GSAP on the Spikingformer 8-768 backbone (default config imagenet.yml)
python imagenet/train.py --model spiking_gsap

# GSAP presets: spiking_gsap_large (MLP ratio 5), spiking_gsap_standard (MLP ratio 4)
python imagenet/train.py --model spiking_gsap_large -c imagenet/imagenet_gsap_large.yml

# GSAP on the QKFormer backbone (qk_gsap_8_384 / qk_gsap_8_512 / qk_gsap_8_768)
python imagenet/train_qk.py --model qk_gsap_8_512 --data_path ./data/imagenet --output_dir ./runs

# Evaluation
python imagenet/test.py --model spiking_gsap --resume /path/to/checkpoint.pth.tar
python imagenet/eval_qk.py --model qk_gsap_8_512 --resume /path/to/checkpoint.pth
```

### CIFAR10-DVS / N-Caltech101 / UCF101-DVS

```bash
python cifar10-dvs/train.py --model spiking_gsap --data-path ./data/CIFAR10DVS
python ncal/train.py        --model spiking_gsap --data-path ./data/NCaltech101
python ucf/train.py         --model spiking_gsap --data-path ./data/UCF101DVS

# Or the provided launcher for N-Caltech101
bash ncal/run_ncal.sh

# Other available --model values: qk_gsap, qk_baseline, sdt_baseline, spikf_baseline, spiking_baseline
```

### Semantic segmentation (ADE20K / Pascal VOC)

The segmentation experiments use a bundled MMSegmentation setup. From the `seg/` directory:

```bash
cd seg

# GSAP on the Spikingformer backbone
python tools/train.py configs/spiking_gsap/fpn_spiking_gsap_ade20k.py
python tools/train.py configs/spiking_gsap/fpn_spiking_gsap_voc12aug.py

# GSAP on the QKFormer backbone
python tools/train.py configs/qk_gsap/fpn_qk_gsap_ade20k.py
python tools/train.py configs/qk_gsap/fpn_qk_gsap_voc12aug.py

# Baselines
python tools/train.py configs/spiking_baseline/fpn_spiking_baseline_ade20k.py
python tools/train.py configs/qk_baseline/fpn_qk_baseline_ade20k.py

# Evaluation / distributed testing
python tools/test.py configs/spiking_gsap/fpn_spiking_gsap_ade20k.py /path/to/checkpoint.pth
bash tools/dist_test.sh configs/spiking_gsap/fpn_spiking_gsap_ade20k.py /path/to/checkpoint.pth 8
```

Dataset roots are set in `seg/configs/_base_/datasets/*.py` (default `data/ADEChallengeData2016`, `data/VOCdevkit/VOC2012`, relative to `seg/`).

### Ablations (CIFAR-10 / CIFAR-100)

```bash
# Single run; --ablation in {full, local_only, large_kernel, mixer_only, no_gate}
python ablation/train.py --model spiking_ablation --ablation no_gate --data-dir ./data/cifar10
python ablation/train.py --model qk_ablation      --ablation local_only --data-dir ./data/cifar100

# Full sweep (2 backbones x 4 ablation variants)
bash ablation/run_all.sh cifar10 ./data/cifar10

# CPU consistency checks (no dataset or GPU required)
python ablation/check_models.py
```

`full` is the intact GSAP attention; the other variants remove or restrict one branch (local path, axial mixer, gate, or kernel size).

### Input encoding experiments (CIFAR)

```bash
# --model in {spiking_gsap, qk_gsap, spiking_baseline}
# --encode-type in {direct, ttfs, rate, phase}
# --backend in {cupy, torch}
python encoder/train.py --model spiking_gsap --encode-type ttfs --backend torch

# CPU consistency checks
python encoder/check_models.py
```

## Outputs

Training logs and checkpoints are written to `./runs/<experiment>_<timestamp>/` by default (override with `--output` / `--output-dir`). Optional Weights & Biases logging is enabled with `--log-wandb` (project name defaults to `GSAP`).

## License

See [LICENSE](LICENSE).
