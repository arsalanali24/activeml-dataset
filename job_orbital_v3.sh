#!/bin/bash
#SBATCH --job-name=orb_v3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/orb_v3_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/orb_v3_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p /pc2/users/h/hpcmual/activeml/logs
mkdir -p /pc2/users/h/hpcmual/activeml/data/orbital_features_v3

python ~/activeml/scripts/gen_orbital_features_v3.py \
    ${SLURM_ARRAY_TASK_ID} 32
