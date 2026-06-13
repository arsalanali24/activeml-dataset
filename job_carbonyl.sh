#!/bin/bash
# job_carbonyl.sh
#SBATCH --job-name=carbonyl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/carbonyl_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/carbonyl_%A_%a.err
#SBATCH --array=0-49

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/generated_carbonyl

python3 -u ~/activeml/scripts/gen_carbonyl.py ${SLURM_ARRAY_TASK_ID}
