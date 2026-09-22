#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python train.py --model spiking_gsap --experiment ncal_spiking_gsap "$@"
