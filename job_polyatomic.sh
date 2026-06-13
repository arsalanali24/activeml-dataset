#!/bin/bash
#SBATCH --job-name=poly
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/poly_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/poly_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p /pc2/users/h/hpcmual/activeml/logs
mkdir -p /pc2/users/h/hpcmual/activeml/data/generated_polyatomic

python ~/activeml/scripts/gen_polyatomic.py ${SLURM_ARRAY_TASK_ID}
