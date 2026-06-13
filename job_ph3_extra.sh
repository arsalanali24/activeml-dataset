#!/bin/bash
#SBATCH --job-name=ph3_extra
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/ph3_extra_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/ph3_extra_%A_%a.err
#SBATCH --array=0-16

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

python3 -u /pc2/users/h/hpcmual/activeml/scripts/worker_ph3_extra.py ${SLURM_ARRAY_TASK_ID}