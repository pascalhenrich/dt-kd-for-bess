#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --time=01:00:00
#SBATCH --mail-user=upsws@student.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=output/kd/slurm/%x/%a/stdout.out
#SBATCH --error=output/kd/slurm/%x/%a/stderr.err



module load devel/python/3.13.3-gnu-14.2

python -m venv .venv
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt

building_id=(13 20 33 35 74 75 82 87 88 101 106 109 130 144 152 153 157 161 169 176)
tr_val=(34 156 90 213 68 -4 104 15 30 22 97 148 39 44 166 13 -6 198 14 -9)
tr_test=(182 358 133 315 206 71 265 52 53 75 173 212 77 96 315 41 316 363 51 16)

cd src

python main.py \
  building_id=${building_id[${SLURM_ARRAY_TASK_ID}]} \
  component=kd \
  component.target_return.val=${tr_val[${SLURM_ARRAY_TASK_ID}]} \
  component.target_return.test=${tr_test[${SLURM_ARRAY_TASK_ID}]} \
  device=cuda
  
