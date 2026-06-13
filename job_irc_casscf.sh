#!/bin/bash
#SBATCH --job-name=irc_casscf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/irc_casscf_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/irc_casscf_%A_%a.err
#SBATCH --array=0-11

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/activeml/logs

python3 -u ~/activeml/scripts/gen_irc_casscf.py ${SLURM_ARRAY_TASK_ID}
