#!/bin/bash
#SBATCH -n 1
#SBATCH -o sacct_sherlock_out_normal.out
#SBATCH -e sacct_sherlock_out_normal.err
#SBATCH -p normal
#SBATCH --mem-per-cpu=4g
#
# can we pass "\t" correctly? maybe for now, just use the HPC standard "|" as a delimiter...
STARTTIME="2019-11-01"
ENDTIME="2020-11-30"
FORMAT="User,Group,GID,Jobname,JobID,JobIDRaw,partition,state,time,ncpus,nnodes,Submit,Eligible,start,end,elapsed,SystemCPU,UserCPU,TotalCPU,NTasks,CPUTimeRaw,Suspended,ReqTRES,AllocTRES"
#FORMAT="ALL"
PARTITION="normal"
#
sacct  --partition=normal  --delimiter="|"  -p --allusers --starttime=2022-03-01T00:00:00 --endtime=2022-04-01T00:00:00 --format=${FORMAT}

sacct  --noheader --partition=normal  --delimiter="|"  -p --allusers --starttime=2022-02-01T00:00:00 --endtime=2022-03-01T00:00:00 --format=${FORMAT}
sacct  --noheader --partition=normal  --delimiter="|"  -p --allusers --starttime=2022-01-01T00:00:00 --endtime=2022-02-01T00:00:00 --format=${FORMAT}
sacct  --noheader --partition=normal  --delimiter="|"  -p --allusers --starttime=2021-12-01T00:00:00 --endtime=2022-01-01T00:00:00 --format=${FORMAT}
sacct  --noheader --partition=normal  --delimiter="|"  -p --allusers --starttime=2021-11-01T00:00:00 --endtime=2021-12-01T00:00:00 --format=${FORMAT}
sacct  --noheader --partition=normal  --delimiter="|"  -p --allusers --starttime=2021-10-01T00:00:00 --endtime=2021-11-01T00:00:00 --format=${FORMAT}

