#!/bin/bash
#SBATCH --job-name=orb_real
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/orb_real_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/orb_real_%A_%a.err
#SBATCH --array=0-15

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/orbital_features_real

python3 ~/activeml/scripts/gen_orbital_features_real.py \
    ${SLURM_ARRAY_TASK_ID} 16
