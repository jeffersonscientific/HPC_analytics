#!/bin/bash
#SBATCH -n 1
#SBATCH -o data/sacct_mazama_out_%j.out
#SBATCH -e data/sacct_mazama_out_%j.err

# can we pass "\t" correctly? maybe for now, just use the HPC standard "|" as a delimiter...
<<<<<<< HEAD
#
STARTTIME1="2019-12-01"
STARTTIME2="2020-03-15"
STARTTIME3="2020-06-30"
#
OUT_FILE_ROOT="sacct_mazama_out"
#FORMAT="User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended"
#FORMAT="ALL"
#FORMAT="User,Group,GID,Jobname,JobID,JobIDRaw,partition,state,time,ncpus,nnodes,Submit,Eligible,start,end,elapsed,SystemCPU,UserCPU,TotalCPU,NTasks,CPUTimeRaw,Suspended"
FORMAT="User,Group,GID,Jobname,JobID,JobIDRaw,partition,state,time,ncpus,nnodes,Submit,Eligible,start,end,elapsed,SystemCPU,UserCPU,TotalCPU,NTasks,CPUTimeRaw,Suspended,ReqGRES,AllocGRES"
#
#srun sacct -a -p --delimiter="|" --${STARTTIME} --format=User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended
#
#srun sacct -a -p --delimiter="|" --starttime=${STARTTIME1} --endtime="2020-06-30" --format=${FORMAT}

sacct -a -p --delimiter="|" --starttime="2019-12-01" --endtime="2019-12-15" --format=${FORMAT}
sacct -a -p --noheader --delimiter="|" --starttime="2019-12-15" --endtime="2020-01-15" --format=${FORMAT}

sacct -a -p --noheader --delimiter="|" --starttime="2020-01-15" --endtime="2020-02-15" --format=${FORMAT}
sacct -a -p --noheader --delimiter="|" --starttime="2020-02-15" --endtime="2020-03-15" --format=${FORMAT}
sacct -a -p --noheader --delimiter="|" --starttime="2020-03-15" --endtime="2020-04-15" --format=${FORMAT}
sacct -a -p --noheader --delimiter="|" --starttime="2020-04-15" --endtime="2020-05-15" --format=${FORMAT}
sacct -a -p --noheader --delimiter="|" --starttime="2020-05-15" --endtime="2020-06-15" --format=${FORMAT}
sacct -a -p --noheader --delimiter="|" --starttime="2020-06-15" --endtime="2020-07-15" --format=${FORMAT}

