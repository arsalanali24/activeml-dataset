#!/bin/bash
#SBATCH --job-name=fecl4_m3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fecl4_m3_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fecl4_m3_%j.err

module purge
module load chem
module load OpenMolcas/25.10-intel-2022a-DMRG

export MOLCAS_NPROCS=16
export MOLCAS_MEM=30000
export MOLCAS_SCRATCH=/scratch/hpc-prf-qehpc/hpcmual/dmrg_3_${SLURM_JOB_ID}
mkdir -p $MOLCAS_SCRATCH

echo "Starting FeCl4 mult=3 at $(date)"
cd /pc2/users/h/hpcmual/activeml/scratch/fecl4_mult3
pymolcas fecl4_mult3.input -f
echo "Finished at $(date)"
