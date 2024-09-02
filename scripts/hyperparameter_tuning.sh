#!/bin/bash

#SBATCH --job-name Hyperparams
#SBATCH --partition dios
#SBATCH -w atenea
#SBATCH --gres=gpu:1
#SBATCH --mem 5g

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export PYTHONPATH=/mnt/homeGPU/atorres/codigo_TFG/src:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/atorres/TFG/
export TFHUB_CACHE_DIR=.

python src/modeling/hyperparameter_tuning.py
