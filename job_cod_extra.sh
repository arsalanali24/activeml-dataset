#!/bin/bash
# job_cod_extra.sh — Script 2: COD for Ti/V/Zn + 7-coord
#SBATCH --job-name=cod_extra
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/cod_extra_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/cod_extra_%A_%a.err
#SBATCH --array=0-149

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/generated_cod_extra
mkdir -p ~/activeml/data/cod_cifs_extra

python3 ~/activeml/scripts/gen_cod_extra.py ${SLURM_ARRAY_TASK_ID}
