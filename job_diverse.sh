#!/bin/bash
#SBATCH --job-name=ms_div
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=03:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/div_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/div_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

echo "Job $SLURM_ARRAY_TASK_ID starting at $(date)"
python ~/activeml/scripts/gen_diverse.py $SLURM_ARRAY_TASK_ID
echo "Done at $(date)"
