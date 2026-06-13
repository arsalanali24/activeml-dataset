#!/bin/bash
#SBATCH --job-name=orb_v2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=0-63
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/orb_v2_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/orb_v2_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

mkdir -p /pc2/users/h/hpcmual/activeml/logs
mkdir -p /pc2/users/h/hpcmual/activeml/data/orbital_features_v2

python ~/activeml/scripts/gen_orbital_features_v2.py \
    ${SLURM_ARRAY_TASK_ID} 64
