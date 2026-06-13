#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# job_4d5d.sh  —  SLURM array job for Phase 1 4d/5d metal generation
#
# HOW TO SUBMIT:
#   1. First count total jobs:
#        python ~/activeml/scripts/gen_4d5d_metals.py --count
#
#   2. Submit (replace NJOBS with the number from step 1, minus 1):
#        sbatch --array=0-NJOBS%150 ~/activeml/scripts/job_4d5d.sh
#
#      Example if --count returns 2400:
#        sbatch --array=0-2399%150 ~/activeml/scripts/job_4d5d.sh
#
#      %150 = run at most 150 jobs simultaneously (polite to other users)
#
# MONITOR:
#   watch -n 60 "squeue -u $USER | grep 4d5d | wc -l && \
#                ls ~/activeml/data/generated_4d5d/*.json | wc -l"
#
# CHECK RESULTS WHEN DONE:
#   python ~/activeml/scripts/check_4d5d_results.py
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=4d5d
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/4d5d_%a_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/4d5d_%a_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p /pc2/users/h/hpcmual/activeml/logs

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"
echo "Script: ~/activeml/scripts/gen_4d5d_metals.py"

python ~/activeml/scripts/gen_4d5d_metals.py ${SLURM_ARRAY_TASK_ID}
EXIT_CODE=$?

echo "Job ${SLURM_ARRAY_TASK_ID} finished exit=${EXIT_CODE} at $(date)"
exit ${EXIT_CODE}
