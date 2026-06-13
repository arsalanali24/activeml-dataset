#!/bin/bash
#SBATCH --job-name=bat4_csd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/bat4_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/bat4_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

echo "Job $SLURM_ARRAY_TASK_ID starting at $(date)"
python ~/activeml/scripts/gen_csd_real.py $SLURM_ARRAY_TASK_ID
echo "Done at $(date)"
