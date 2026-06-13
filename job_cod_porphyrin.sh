#!/bin/bash
# job_cod_porphyrin.sh
#SBATCH --job-name=cod_porph
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=14:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/cod_porph_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/cod_porph_%A_%a.err
#SBATCH --array=0-299

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/generated_cod_porphyrin

python3 -u ~/activeml/scripts/gen_cod_phosphine_porphyrin.py \
    run_porphyrin ${SLURM_ARRAY_TASK_ID}
