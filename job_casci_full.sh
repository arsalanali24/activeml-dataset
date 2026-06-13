#!/bin/bash
#SBATCH --job-name=casci_f
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/casfull_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/casfull_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4

CAS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" \
    /pc2/users/h/hpcmual/activeml/scripts/casci_full_commands.txt)
echo "Job $SLURM_ARRAY_TASK_ID at $(date)"
python ~/activeml/scripts/casci_full_single.py $CAS
echo "Done at $(date)"
