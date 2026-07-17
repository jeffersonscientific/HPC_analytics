#!/bin/bash
#
#SBATCH --job-name=pptx_partition_report
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6g
#SBATCH --time=04:00:00
#SBATCH --partition=serc
#
module purge
module load py-sherlock/3.9
#
SRC_DIR=$HOME/Codes/sdss_hpc_analytics
#
PPTX_PARTITION=serc
REPORT_LEN=180
END_DT=2026-07-16
GROUP=None
USER=None
NCPU=$SLURM_CPUS_ON_NODE
#DELIM_SACCT="*"

#python pptx_reports.py --report_type=partition --partition=$PPTX_PARTITION --end_dtm=$END_DT --report_len_days=$REPORT_LEN --verbose=True
python pptx_reports.py --help


 
