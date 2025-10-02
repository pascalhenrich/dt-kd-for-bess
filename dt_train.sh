#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition=cpu
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

python main.py \
  name=train_new \
  building_id=13 \
  component=dt \
  component.transformer.num_heads=4 \
  component.transformer.num_layers=6\
  component.max_context_length=96 \
  component.model_dim=512
