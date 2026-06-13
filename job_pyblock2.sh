#!/bin/bash
#SBATCH --job-name=fecl4_pb2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fecl4_pb2_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fecl4_pb2_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

echo "Starting pyblock2 DMRG at $(date)"
python ~/activeml/scratch/fecl4_mult5/fecl4_pyblock2.py
echo "Finished at $(date)"
