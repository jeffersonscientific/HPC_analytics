#!/bin/python3
#SBATCH --output=calc_sacct_obj.out
#SBATCH --error=calc_sacct_obj.err
#SBATCH --mem=8gb
#SBATCH --ntasks=4
#SBATCH -N 1

import matplotlib
matplotlib.use("Agg")

import numpy
import scipy
import matplotlib.dates as mpd
import pylab as plt
import datetime as dtm
import pytz
import multiprocessing as mpp
import pickle
import os
import subprocess
#
# TODO: phase out unreferenced hpc_lib calls...
import hpc_lib

#
data_file_name = 'data/sacct_owners_out_3500489.out'
#data_file_name = 'data/sacct_mazama_0623_tool8.out'
#data_file_name = '/scratch/myoder96/HPC_analytics/data/sacct_serc_20200622.out'
#
pickle_in=False
pickle_out=True      # NOTE: will only be effective if pickle_in==False
#
data_path, data_fname = os.path.split(data_file_name)
data_fname_root, data_fname_ext = os.path.splitext(data_file_name)
#
pkl_name = "{}.pkl".format(os.path.splitext(data_file_name)[0])
tex_fname = '{}.tex'.format(data_fname_root)
#
n_cpu=8
#
# temporarily, comment these out so we can just run the report:
#sacct_mazama = hpc_lib.SACCT_data_handler(data_file_name=data_file_name)
##
#with open(pkl_name, 'wb') as fout:
#    #
#    out_pkl = pickle.dump(sacct_mazama, fout)
#
# and we can script the report if we like:
#TODO: move variable definitions to top and integrate workflow.

#
if pickle_in:
	print('*** Pickle_IN: Skipping SACCT_data_handler() compute')
	with open(pkl_name, 'rb') as fin:
		sacct_mazama = pickle.load(fin)
else:
	sacct_mazama = hpc_lib.SACCT_data_handler(data_file_name=data_file_name, n_cpu=n_cpu)
	#
	if pickle_out:
		with open(pkl_name, 'wb') as fout:
			 out_pkl = pickle.dump(sacct_mazama, fout)
		#
	#
#

##system_name='mazama_0623'
#system_name=data_fname_root
##groups_fname='mazama_groups.json'
#groups_fname={'ALL':list(set(sacct_mazama.jobs_summary['User']))}
#output_path = 'output/{}_HPC_analytics'.format(system_name)
#print('*** cleaning up old out_path...')
#sp_status = subprocess.run(['rm', '-rf', output_path])
#print('*** out_path (should be) removed: ', sp_status)
#
#print('*** Compute report:')
#mazama_report = hpc_lib.SACCT_groups_analyzer_report(out_path=output_path, groups=groups_fname,
#                                            tex_filename=tex_fname,
#                                            SACCT_obj=sacct_mazama, max_rws=None)

