#!/bin/bash
#SBATCH --job-name=fix_irc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fix_irc_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fix_irc_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

mkdir -p ~/activeml/logs

python3 -u ~/activeml/scripts/fix_irc_snapshots.py
