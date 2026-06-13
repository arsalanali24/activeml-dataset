#!/bin/bash
#SBATCH --job-name=bench_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/bench_test_%A_%a.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/bench_test_%A_%a.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/activeml/logs
mkdir -p ~/activeml/data/benchmark_test_cases

python3 -u ~/activeml/scripts/gen_benchmark_test.py ${SLURM_ARRAY_TASK_ID}
