#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition=cpu
#SBATCH --array=0-7
#SBATCH --time=01:00:00
#SBATCH --mail-user=upsws@student.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=output/ddpg/slurm/%a/stdout.out
#SBATCH --error=output/ddpg/slurm/%a/stderr.err

module load devel/python/3.13.3-gnu-14.2

python -m venv .venv
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt


sizes=(144)
offsets=(24)

# Compute index combinations
size_index=$(( SLURM_ARRAY_TASK_ID / ${#offsets[@]} ))
offset_index=$(( SLURM_ARRAY_TASK_ID % ${#offsets[@]} ))

sliding_window_size=${sizes[$size_index]}
sliding_window_offset=${offsets[$offset_index]}

cd src

python main.py \
    building_id=13 \
    component=ddpg \
    component.mode=train_full \
    component.dataset.sliding_window_size=$sliding_window_size \
    component.dataset.sliding_window_offset=$sliding_window_offset