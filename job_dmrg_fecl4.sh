#!/bin/bash
#SBATCH --job-name=dmrg_fecl4
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/%x_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/%x_%j.err

module purge
module load chem
module load OpenMolcas/25.10-intel-2022a-DMRG

export MOLCAS_NPROCS=16
export MOLCAS_MEM=30000
export MOLCAS_SCRATCH=$HOME/activeml/scratch/dmrg_${SLURM_JOB_ID}
mkdir -p $MOLCAS_SCRATCH

echo "DMRG job started: $(date)"
cd ~/activeml/scratch/fecl4_quintet
pymolcas fecl4_quintet.input -f
echo "DMRG job finished: $(date)"
