#!/bin/bash
#SBATCH -n 1
#SBATCH -o sacct_sherlock_out_serc.out
#SBATCH -e sacct_sherlock_out_serc.err
#SBATCH --partition=serc
#
# can we pass "\t" correctly? maybe for now, just use the HPC standard "|" as a delimiter...
STARTTIME="2019-11-01"
ENDTIME="2020-11-30"
#FORMAT="User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended"
# old-school format list elements for jobarrayid, stepid, arraytaskid (i think, maybe not in that order)
# -o %F, %i, %K,
# note, little "-o", but it looks like -o, -O, --Format, --format are all cross-supported ? or at least partially? It also looks like the field names are not cap-sensitive?
FORMAT="User,Group,GID,Jobname,JobID,JobIDRaw,partition,state,time,ncpus,nnodes,Submit,Eligible,start,end,elapsed,SystemCPU,UserCPU,TotalCPU,NTasks,CPUTimeRaw,Suspended,ReqTRES,AllocTRES"
#,MaxRss,MaxVMSize
#
#FORMAT="ALL"
PARTITION="serc"
#
#sacct -a -p --delimiter="|" --${STARTTIME} --format=User,Group,GID,JobID,Jobname,partition,state,Submit,time,Eligible,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,SystemCPU,UserCPU,TotalCPU,Suspended

sacct --allusers --partition=$PARTITION -p --delimiter="|" --starttime=2020-11-15 --endtime=2020-12-15 --format=${FORMAT}
sacct --allusers --partition=$PARTITION -p --noheader --delimiter="|" --starttime=2020-12-15 --endtime=2021-01-15 --format=${FORMAT}
sacct --allusers --partition=$PARTITION -p --noheader --delimiter="|" --starttime=2021-01-15 --endtime=2021-02-15 --format=${FORMAT}
sacct --allusers --partition=$PARTITION -p --noheader --delimiter="|" --starttime=2021-02-15 --endtime=2021-03-15 --format=${FORMAT}
sacct --allusers --partition=$PARTITION -p --noheader --delimiter="|" --starttime=2021-03-15 --endtime=2021-04-15 --format=${FORMAT}
sacct --allusers --partition=$PARTITION -p --noheader --delimiter="|" --starttime=2021-04-15 --endtime=2021-05-15 --format=${FORMAT}
#
#sacct --allusers --partition=hns -p --delimiter="|" --starttime=${STARTTIME} --endtime=${ENDTIME} --format=${FORMAT}
