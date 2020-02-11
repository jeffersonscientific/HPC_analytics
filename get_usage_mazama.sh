#!/bin/bash
#SBATCH -n 1
#SBATCH -o data/sacct_mazama_out.out
#SBATCH -e data/sacct_mazama_out.err

# can we pass "\t" correctly? maybe for now, just use the HPC standard "|" as a delimiter...
STARTTIME="2019-01-01"
#FORMAT="User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended"
FORMAT="ALL"
#
#srun sacct -a -p --delimiter="|" --${STARTTIME} --format=User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended

srun sacct -a -p --delimiter="|" --starttime=${STARTTIME} --format=${FORMAT}

