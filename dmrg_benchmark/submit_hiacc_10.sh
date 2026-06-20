#!/bin/bash
#SBATCH --job-name=dmrg_hiacc
#SBATCH --output=/scratch/hpc-prf-qehpc/hpcmual/dmrg_labels/logs/hiacc_%A_%a.out
#SBATCH --error=/scratch/hpc-prf-qehpc/hpcmual/dmrg_labels/logs/hiacc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=normal
#SBATCH --array=1-10%5

BLOCK2ENV="/pc2/users/h/hpcmual/envs/block2env"
source "${BLOCK2ENV}/bin/activate"
module load numlib/imkl/2023.2.0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

INDEX="/pc2/users/h/hpcmual/activeml/dmrg_labels/benchmark_10_v2_index.txt"
SCRATCH="/scratch/hpc-prf-qehpc/hpcmual/dmrg_hiacc"

SYSTEM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$INDEX")
echo "Job ${SLURM_JOB_ID} Task ${SLURM_ARRAY_TASK_ID} $(hostname) $(date)"
echo "System: $SYSTEM"

python ~/activeml/scripts/run_dmrg_highaccuracy.py \
    --system   "$SYSTEM" \
    --scratch  "$SCRATCH" \
    --n_threads 8

echo "Exit: $?"
