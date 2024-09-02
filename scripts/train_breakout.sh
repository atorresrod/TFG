#!/bin/bash

#SBATCH --job-name Breakout
#SBATCH --partition dios
#SBATCH -w dionisio
#SBATCH --gres=gpu:1
#SBATCH --mem 70g

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export PYTHONPATH=/mnt/homeGPU/atorres/codigo_TFG/src:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/atorres/TFG/
export TFHUB_CACHE_DIR=.

python src/modeling/train_dqn.py "BreakoutNoFrameskip-v4" "saves/models/breakout/dqn.pth" "saves/metrics/breakout/dqn.pth" --progress_path "saves/progress/breakout/dqn.pth"
