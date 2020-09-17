#!/bin/bash
#
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --output=comp_sact_obj_%j.out
#SBATCH --error=comp_sacct_obj_%j.err
#SBATCH --time=36:00:00
#
# sbatch -n 4 --mem=16gb --output=serc_calc_obj.out --error=serc_calc_obj.err compute_sacct_obj.py -f data/serc_usage_20200914.out -p 0 -q 1 -n 4
#
module purge
module load python/3
module load py-h5py/2.10.0_py36
module load viz
module load py-matplotlib/3.2.1_py36
# module load py-ipython/6.1.0_py36
#
module list
#
SACCT_FILE="data/serc_usage_20200917.out"
#
srun --ntasks=${SLURM_NTASKS} --mem-per-cpu=${SLURM_MEM_PER_CPU} --cpus-per-task=${SLURM_CPUS_PER_TASK} python3 compute_sacct_obj.py -f${SACCT_FILE} -p0 -q1 -n${SLURM_CPUS_PER_TASK}
