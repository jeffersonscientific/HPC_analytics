#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --time=10:00:00
#SBATCH -o data/process_myparts.out
#SBATCH -e data/process_myparts.err
#
module load python/3.6.1
module load py-ipython/6.1.0_py36
#module load py-scipystack/1.0_py36
# module load py-scipy py-numpy py-matplotlib py-pandas
#module load py-scipy/1.1.0_py36
#module load py-numpy/1.17.2_py36
#
#module load viz
#module load py-matplotlib/3.1.1_py36
module load py-pandas/0.23.0_py36

#
DATA_FILE="sacct_sherlock_out_myparts.out"
#DATA_FILE="data/demo_data.out"
cd $SCRATCH/HPC_analytics
#
echo "srun-ning:: srun -n ${SLURM_NTASKS} -N ${SLURM_NNODES} ipython"
#srun -n 1 ipython -c "import hpc_lib;print('running!')"
#srun -n ${SLURM_NTASKS} -N ${SLURM_NNODES}
ipython -c "import hpc_lib; my_sacct = hpc_lib.SACCT_data_handler(data_file_name='$DATA_FILE', delim='|'); fout = open('data/sacct_myparts_summary.pkl', 'wb'); out_pk = hpc_lib.pickle.dump(my_sacct, fout); fout.close()"
