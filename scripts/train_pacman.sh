#!/bin/bash

#SBATCH --job-name Pacman
#SBATCH --partition dios
#SBATCH -w hera
#SBATCH --gres=gpu:1
#SBATCH --mem 70g

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export PYTHONPATH=/mnt/homeGPU/atorres/codigo_TFG/src:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/atorres/TFG/
export TFHUB_CACHE_DIR=.

python src/modeling/train_dqn.py "MsPacmanNoFrameskip-v4" "saves/models/pacman/dqn.pth" "saves/metrics/pacman/dqn.pth" --progress_path "saves/progress/pacman/dqn.pth"
