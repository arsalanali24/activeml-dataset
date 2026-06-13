#!/bin/bash
#SBATCH --job-name=hf_v2
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:40:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/hfv2_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/hfv2_%a_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=4

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" \
    /pc2/users/h/hpcmual/activeml/scripts/hf_v2_commands.txt)
echo "Running: $CMD at $(date)"
python ~/activeml/scripts/pyscf_features_v2.py $CMD
echo "Done at $(date)"
