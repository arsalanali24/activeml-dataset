#!/bin/bash
# job_ml2_features.sh
# Run ML2 feature extraction with UHF rerun (accurate but slow)
# For fast estimation without UHF: run estimate_all on login node instead
#SBATCH --job-name=ml2_feat
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/ml2_feat_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/ml2_feat_%A_%a.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/ml2_features

python3 -u ~/activeml/scripts/gen_ml2_features.py ${SLURM_ARRAY_TASK_ID}
