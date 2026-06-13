#!/bin/bash
#SBATCH --job-name=hf_mix
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:40:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/hfmix_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/hfmix_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4

CAS_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" \
    /pc2/users/h/hpcmual/activeml/scripts/hf_mixed_commands.txt)
echo "Running: $CAS_PATH at $(date)"
python ~/activeml/scripts/pyscf_features_mixed.py $CAS_PATH
echo "Done at $(date)"
