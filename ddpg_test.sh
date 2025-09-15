#!/bin/sh
#SBATCH --job-name=ddpg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-user=upsws@student.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=output/%x_%a/stdout.out
#SBATCH --error=output/%x_%a/stderr.err


module load devel/python/3.13.3-gnu-14.2

python -m venv .venv
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt

cd src

python main.py name=$SLURM_JOB_NAME device=cuda