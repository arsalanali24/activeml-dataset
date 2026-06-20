#!/bin/bash
#SBATCH --job-name=asf_all
#SBATCH --output=/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch/asf_all_%j.out
#SBATCH --error=/scratch/hpc-prf-qehpc/hpcmual/dmrg_scratch/asf_all_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --qos=cont
#SBATCH --nodelist=n2cn0203

echo "Started: $(date)"
source /pc2/users/h/hpcmual/envs/block2env/bin/activate
export LD_LIBRARY_PATH=/pc2/users/h/hpcmual/envs/block2env/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd /pc2/users/h/hpcmual/activeml/scripts

python asf_all_cases.py

echo "Finished: $(date)"
