#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition=cpu
#SBATCH --array=0-19
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

building_id=(13 20 33 35 74 75 82 87 88 101 106 109 130 144 152 153 157 161 169 176)

cd src

python main.py name=${SLURM_JOB_NAME} building_id=${building_id[${SLURM_ARRAY_TASK_ID}]}

mv "../output/ddpg/slurm/${SLURM_ARRAY_TASK_ID}/stdout.out" "../output/ddpg/${building_id}/stdout.out"
mv "../output/ddpg/slurm/${SLURM_ARRAY_TASK_ID}/stderr.err" "../output/ddpg/${building_id}/stderr.err"