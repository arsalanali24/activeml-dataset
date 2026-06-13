#!/bin/bash
# job_bidentate.sh — Script 3: bidentate CASSCF(14,14) Paper 2
#SBATCH --job-name=bidentate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/bidentate_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/bidentate_%A_%a.err
#SBATCH --array=0-399

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/generated_bidentate

python3 ~/activeml/scripts/gen_bidentate.py ${SLURM_ARRAY_TASK_ID}
