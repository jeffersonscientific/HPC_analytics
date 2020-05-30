#!/bin/bash
#SBATCH -n 1
#SBATCH -o data/sacct_mazama_out.out
#SBATCH -e data/sacct_mazama_out.err

# can we pass "\t" correctly? maybe for now, just use the HPC standard "|" as a delimiter...
STARTTIME1="2019-10-15"
STARTTIME2="2019-12-15"
STARTTIME3="2020-04-01"

OUT_FILE_ROOT="sacct_mazama_out"
#FORMAT="User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended"
#FORMAT="ALL"
#FORMAT="User,Group,GID,Jobname,JobID,JobIDRaw,partition,state,time,ncpus,nnodes,Submit,Eligible,start,end,elapsed,SystemCPU,UserCPU,TotalCPU,NTasks,CPUTimeRaw,Suspended"
FORMAT="User,Group,GID,Jobname,JobID,JobIDRaw,partition,state,time,ncpus,nnodes,Submit,Eligible,start,end,elapsed,SystemCPU,UserCPU,TotalCPU,NTasks,CPUTimeRaw,Suspended"
#
#srun sacct -a -p --delimiter="|" --${STARTTIME} --format=User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended
#
sacct -a -p --delimiter="|" --starttime=${STARTTIME1} --format=${FORMAT}

