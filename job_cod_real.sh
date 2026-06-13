#!/bin/bash
#SBATCH --job-name=cod_real
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/cod_real_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/cod_real_%A_%a.err
#SBATCH --array=0-216

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/generated_cod

python3 ~/activeml/scripts/gen_cod_real.py ${SLURM_ARRAY_TASK_ID}
