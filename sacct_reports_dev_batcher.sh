#!/bin/bash
#
#SBATCH --job-name=sacct_reports_dev
#SBATCH --partition=serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --output=sacct_report_batcher_%j.out
#SBATCH --error=sacct_report_batcher_%j.out
#SBATCH --time=1:00:00
#
module purge
module load anaconda-cees-beta/
module load system texlive/
#
CMD="python SACCT_reports_dev.py"
#
echo "*** DEBUG: Execute script: ${CMD}"
#
$CMD
#
echo "*** DEBUG: Script (hopefully) executed."
#

