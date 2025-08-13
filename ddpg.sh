#!/bin/sh
#SBATCH --job-name=ddpg
#SBATCH --partition=cpu
#SBATCH --array=0-79                
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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

# Lists
forecast_horizons=(1 12 24 32 48)
batch_sizes=(32 64 128 256)
frames_per_batch=(50 100 200 400)

# Lengths
len_fh=${#forecast_horizons[@]}     # 5
len_bs=${#batch_sizes[@]}           # 4
len_fp=${#frames_per_batch[@]}      # 4

# Decode SLURM_ARRAY_TASK_ID
index=$SLURM_ARRAY_TASK_ID

i_fh=$(( index / (len_bs * len_fp) ))
rem=$(( index % (len_bs * len_fp) ))

i_bs=$(( rem / len_fp ))
i_fp=$(( rem % len_fp ))

# Get parameter values
fh=${forecast_horizons[$i_fh]}
bs=${batch_sizes[$i_bs]}
fp=${frames_per_batch[$i_fp]}

FULL_JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}

python -m experiments.ddpg.train_eval name=$FULL_JOB_NAME forecast_horizon=$fh batch_size=$bs frames_per_batch=$fp