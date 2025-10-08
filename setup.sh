#!/bin/sh
#SBATCH --job-name=setup
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --mail-user=upsws@student.kit.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=output/setup/stdout.out
#SBATCH --error=output/setup/stderr.err

module load devel/python/3.13.3-gnu-14.2

python -m venv .venv
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi
pip install --upgrade pip
pip install --upgrade -r requirements.txt