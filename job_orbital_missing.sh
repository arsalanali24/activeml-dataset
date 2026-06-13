#!/bin/bash
#SBATCH --job-name=orb_miss
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/orb_miss_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/orb_miss_%A_%a.err
#SBATCH --array=0-2569%200

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4

python3 ~/activeml/scripts/gen_orbital_features.py ${SLURM_ARRAY_TASK_ID} 2570
