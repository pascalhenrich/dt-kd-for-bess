#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition=cpu
#SBATCH --array=0-539
#SBATCH --time=00:30:00
#SBATCH --mail-user=upsws@student.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=output/dt/slurm/%a/stdout.out
#SBATCH --error=output/dt/slurm/%a/stderr.err


module load devel/python/3.13.3-gnu-14.2

python -m venv .venv
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt

cd src

# Hyperparameter grids
HEADS=(1 2 4)
LAYERS=(2 3 4 5 6)
CTX=(24 48 96)
DIMS=(64 128 256 512)
BS=(32 64 128)

# Grid sizes
NH=${#HEADS[@]}   # 3
NL=${#LAYERS[@]}  # 5
NC=${#CTX[@]}     # 3
ND=${#DIMS[@]}    # 4
NB=${#BS[@]}      # 3

# Map array index -> combo (row-major order: heads × layers × ctx × dims × bs)
i=${SLURM_ARRAY_TASK_ID}

b_idx=$(( i % NB ));  i=$(( i / NB ))
d_idx=$(( i % ND ));  i=$(( i / ND ))
c_idx=$(( i % NC ));  i=$(( i / NC ))
l_idx=$(( i % NL ));  i=$(( i / NL ))
h_idx=$(( i % NH ))

NUM_HEADS=${HEADS[$h_idx]}
NUM_LAYERS=${LAYERS[$l_idx]}
MAX_CONTEXT=${CTX[$c_idx]}
MODEL_DIM=${DIMS[$d_idx]}
BATCH_SIZE=${BS[$b_idx]}


python main.py \
  name=${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID} \
  building_id=13 \
  component=dt \
  component.batch_size=${BATCH_SIZE} \
  component.transformer.num_heads=${NUM_HEADS} \
  component.transformer.num_layers=${NUM_LAYERS} \
  component.max_context_length=${MAX_CONTEXT} \
  component.model_dim=${MODEL_DIM}
