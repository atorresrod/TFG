#!/bin/bash

#SBATCH --job-name FrozenLake
#SBATCH --partition dios
#SBATCH -w titan

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export PYTHONPATH=/mnt/homeGPU/atorres/codigo_TFG/src:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/atorres/TFG/
export TFHUB_CACHE_DIR=.

python src/modeling/train_tabular.py "FrozenLake-v1" "saves/models/frozen_lake" "saves/metrics/frozen_lake"

