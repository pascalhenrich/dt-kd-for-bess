#!/bin/sh
#SBATCH --job-name=run0
#SBATCH --partition=cpu
#SBATCH --array=0-728
#SBATCH --time=10:00:00
#SBATCH --mail-user=upsws@student.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=output/ddpg/13/%x_%a/stdout.out
#SBATCH --error=output/ddpg/13/%x_%a/stderr.err


module load devel/python/3.13.3-gnu-14.2

python -m venv .venv
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt

cd src

# Parameter lists (no tau)
forecast_horizons=(24 32 48)
batch_sizes=(64 128 256)
frames_per_batch=(50 100 200)
replay_buffer_size=(10000 100000 1000000)
lr_actor=(0.0001 0.0005 0.001)
lr_value=(0.001 0.005 0.01)

# Lengths
len_fh=${#forecast_horizons[@]}   # 3
len_bs=${#batch_sizes[@]}         # 3
len_fp=${#frames_per_batch[@]}    # 3
len_rb=${#replay_buffer_size[@]}  # 3
len_lra=${#lr_actor[@]}           # 3
len_lrv=${#lr_value[@]}          # 3

# Total combos = 729
total=$((len_fh * len_bs * len_fp * len_rb * len_lra * len_lrv))

# Decode array index
index=${SLURM_ARRAY_TASK_ID:?}
if [ "$index" -ge "$total" ]; then
  echo "Index $index out of range (total=$total)."; exit 1
fi

i_fh=$(( index / (len_bs * len_fp * len_rb * len_lra * len_lrv) ))
rem=$(( index % (len_bs * len_fp * len_rb * len_lra * len_lrv) ))

i_bs=$(( rem / (len_fp * len_rb * len_lra * len_lrv) ))
rem=$(( rem % (len_fp * len_rb * len_lra * len_lrv) ))

i_fp=$(( rem / (len_rb * len_lra * len_lrv) ))
rem=$(( rem % (len_rb * len_lra * len_lrv) ))

i_rb=$(( rem / (len_lra * len_lrv) ))
rem=$(( rem % (len_lra * len_lrv) ))

i_lra=$(( rem / len_lrv ))
i_lrv=$(( rem % len_lrv ))

# Values
fh=${forecast_horizons[$i_fh]}
bs=${batch_sizes[$i_bs]}
fp=${frames_per_batch[$i_fp]}
rb=${replay_buffer_size[$i_rb]}
lra=${lr_actor[$i_lra]}
lrv=${lr_value[$i_lrv]}

FULL_JOB_NAME=${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}

python main.py \
  name="$FULL_JOB_NAME" \
  customer=13 \
  component.dataset.forecast_horizon="$fh" \
  component.batch_size="$bs" \
  component.collector_frames_per_batch="$fp" \
  component.replay_buffer_size="$rb" \
  component.lr.actor="$lra" \
  component.lr.value="$lrv"