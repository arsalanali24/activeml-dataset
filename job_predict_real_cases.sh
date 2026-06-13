#!/bin/bash
#SBATCH --job-name=predict_rc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/predict_rc_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/predict_rc_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH

pip install scikit-learn --user --quiet 2>/dev/null

python3 -u ~/activeml/scripts/predict_real_cases.py
