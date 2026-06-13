#!/bin/bash
#SBATCH --job-name=fecl4_dmrg3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fecl4_dmrg3_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fecl4_dmrg3_%j.err

module purge
module load chem
module load OpenMolcas/25.10-intel-2022a-DMRG

# Set QCMaquis interface manually
export QCMAQUIS_NEW_INTERFACE=/opt/software/pc2/EB-SW/software/QCMaquis/4.0.0-intel-2022a/bin/qcmaquis_maquis
export MOLCAS_NPROCS=16
export MOLCAS_MEM=30000
export MOLCAS_SCRATCH=/scratch/hpc-prf-qehpc/hpcmual/dmrg_v3_${SLURM_JOB_ID}
mkdir -p $MOLCAS_SCRATCH

echo "QCMaquis: $QCMAQUIS_NEW_INTERFACE"
echo "Starting at $(date)"
cd /pc2/users/h/hpcmual/activeml/scratch/fecl4_mult5
pymolcas fecl4_mult5.input -f
echo "Finished at $(date)"
