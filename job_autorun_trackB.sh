#!/bin/bash
#SBATCH --job-name=autorun_B
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/autorun_B_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/autorun_B_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data

python3 ~/activeml/scripts/autorun_trackB.py
