#!/bin/bash
#SBATCH --job-name=dft_irc_pd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/dft_irc_pd_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/dft_irc_pd_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/irc_pd

# Run full IRC pipeline
python3 ~/activeml/scripts/dft_irc_pd.py
