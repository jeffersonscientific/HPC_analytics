#!/bin/bash
#SBATCH -n 1
#SBATCH -o sacct_sherlock_out_hns.out
#SBATCH -e sacct_sherlock_out_hns.err
#SBATCH -p hns
#
# can we pass "\t" correctly? maybe for now, just use the HPC standard "|" as a delimiter...
STARTTIME="2019-11-01"
ENDTIME="2020-11-30"
FORMAT="User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended"
#FORMAT="ALL"
PARTITION="hns"
#
#srun sacct -a -p --delimiter="|" --${STARTTIME} --format=User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended

srun sacct --allusers --partition=$PARTITION -p --delimiter="|" --starttime=2019-08-16 --endtime=2019-09-15 --format=${FORMAT}
srun sacct --allusers --partition=$PARTITION -p --delimiter="|" --starttime=2019-09-16 --endtime=2019-10-15 --format=${FORMAT}
srun sacct --allusers --partition=$PARTITION -p --delimiter="|" --starttime=2019-10-16 --endtime=2019-11-15 --format=${FORMAT}
srun sacct --allusers --partition=$PARTITION -p --delimiter="|" --starttime=2019-11-16 --endtime=2019-12-15 --format=${FORMAT}
srun sacct --allusers --partition=$PARTITION -p --delimiter="|" --starttime=2019-12-16 --endtime=2020-01-15 --format=${FORMAT}
srun sacct --allusers --partition=$PARTITION -p --delimiter="|" --starttime=2020-01-16 --endtime=2020-02-15 --format=${FORMAT}

#srun sacct --allusers --partition=hns -p --delimiter="|" --starttime=${STARTTIME} --endtime=${ENDTIME} --format=${FORMAT}


