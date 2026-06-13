#!/bin/bash
#SBATCH --job-name=fecl4_3s
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fecl4_3s_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fecl4_3s_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

echo "Starting at $(date)"
python ~/activeml/scripts/casscf_three_spins.py
echo "Finished at $(date)"
