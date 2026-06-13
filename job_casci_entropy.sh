#!/bin/bash
#SBATCH --job-name=casci_ent
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/casci_ent_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/casci_ent_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

echo "CASCI entropy test starting at $(date)"
python ~/activeml/scripts/casci_orbital_entropy.py 30
echo "Done at $(date)"
