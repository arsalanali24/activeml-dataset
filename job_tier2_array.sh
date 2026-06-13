#!/bin/bash
#SBATCH --job-name=tier2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=06:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/tier2_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/tier2_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

python ~/activeml/scripts/extract_tier2.py \
    --chunk-index ${SLURM_ARRAY_TASK_ID} \
    --chunk-total 32
