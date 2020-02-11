#!/bin/bash
#SBATCH -n 1
#SBATCH -o sacct_sherlock_out.out
#SBATCH -e sacct_sherlock_out.err

# can we pass "\t" correctly? maybe for now, just use the HPC standard "|" as a delimiter...
STARTTIME="2020-01-01"
ENDTIME="2020-1-31"
#FORMAT="User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended"
FORMAT="ALL"
#
#srun sacct -a -p --delimiter="|" --${STARTTIME} --format=User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended

srun sacct --allusers --partition=hns -p --delimiter="|" --starttime=${STARTTIME} --format=${FORMAT}

