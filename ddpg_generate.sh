#!/bin/sh
#SBATCH --job-name=generate
#SBATCH --partition=cpu
#SBATCH --array=0-19
#SBATCH --time=00:10:00
#SBATCH --mail-user=upsws@student.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=output/ddpg/slurm/%x/%a/stdout.out
#SBATCH --error=output/ddpg/slurm/%x/%a/stderr.err


module load devel/python/3.13.3-gnu-14.2

python -m venv .venv
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

building_id=(13 20 33 35 74 75 82 87 88 101 106 109 130 144 152 153 157 161 169 176)

cd src

python main.py \
  component=ddpg \
  component.mode=generate \
  model_path=../model/ddpg/train_half/${building_id[${SLURM_ARRAY_TASK_ID}]} \
  building_id=${building_id[${SLURM_ARRAY_TASK_ID}]} \
  component.dataset.sliding_window_size=1344 \
  component.dataset.sliding_window_offset=1344
