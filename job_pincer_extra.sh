#!/bin/bash
#SBATCH --job-name=pincer_extra
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=14:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/pincer_extra_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/pincer_extra_%A_%a.err
#SBATCH --array=0-8

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

python3 -u /pc2/users/h/hpcmual/activeml/scripts/worker_pincer_extra.py ${SLURM_ARRAY_TASK_ID}