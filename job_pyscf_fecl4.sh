#!/bin/bash
#SBATCH --job-name=pyscf_fecl4
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/%x_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/%x_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0

export OMP_NUM_THREADS=8
export PATH=$HOME/.local/bin:$PATH

echo "PySCF job started: $(date)"
python ~/activeml/scripts/pyscf_features.py \
    --metal Fe --ligand Cl --nlig 4 \
    --charge -2 --spin 4
echo "PySCF job finished: $(date)"
