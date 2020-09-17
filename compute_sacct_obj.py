

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
import sys, getopt
#
# TODO: phase out unreferenced hpc_lib calls...
import hpc_lib
#
def main(data_file_name='data/sacct_owners_out_3500489.out', pickle_in=True, pickle_out=True, n_cpu=8, max_rows=10000):
    #
    #data_file_name = 'data/sacct_owners_out_3500489.out'
    #data_file_name = 'data/sacct_mazama_0623_tool8.out'
    #data_file_name = '/scratch/myoder96/HPC_analytics/data/sacct_serc_20200622.out'
    #
    #pickle_in=False
    #pickle_out=True      # NOTE: will only be effective if pickle_in==False
    #
    # TODO: compute n_cpu from SLURM variables (
    #n_cpu=12
    ######################
    #
    data_path, data_fname = os.path.split(data_file_name)
    data_fname_root, data_fname_ext = os.path.splitext(data_file_name)
    #
    pkl_name = "{}.pkl".format(os.path.splitext(data_file_name)[0])
    tex_fname = '{}.tex'.format(data_fname_root)
    hdf5_name = '{}.h5'.format(data_fname_root)
    #
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
    print('*** executing: ', locals().items() )
    #
    #return None
    #sys.exit(2)
    #
    if pickle_in:
        print('*** Pickle_IN: Skipping SACCT_data_handler() compute')
        with open(pkl_name, 'rb') as fin:
            sacct_mazama = pickle.load(fin)
    else:
        print('** Building new SACCT_data_handler() from: {}'.format(data_file_name))
        sacct_mazama = hpc_lib.SACCT_data_handler(data_file_name=data_file_name, n_cpu=n_cpu, max_rows=max_rows)
        #
	#
        if pickle_out:
            print('** Writing pickle() object to: {}'.format(pkl_name))
            with open(pkl_name, 'wb') as fout:
                 out_pkl = pickle.dump(sacct_mazama, fout)
            #
    # For now, piggyback the HDF5 option with pickle_out. Eventually,
    #  replace pickle() or separate these.
    if pickle_out:
        print('** Writing HDF5() object to: {}'.format(hdf5_name))
        # TODO: move this, or add a parameter, do_h5, or just get rid of pickle. with h5, we don't really need it.
        h5out=sacct_mazama.write_hdf5(h5out_file=hdf5_name)
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
#
if __name__ == '__main__':
    #
    print('*** argv: [{}] {} **'.format(len(sys.argv), sys.argv))
    prams = {}
    
    if len(sys.argv)>1:
        #print('*** getting args...')
        opts,args = getopt.getopt(sys.argv[1:], 'f:p:q:n:r:', ['data_file_name=', 'pickle_in=', 'pickle_out=', 'n_cpu='])
        print('*** *** ', opts, args)
        #
        for opt,arg in opts:
            #print('** * **: opt, arg: ({}, {})'.format(opt, arg))
            if opt in ('-f', '--data_file_name'):
                data_file_name=arg
                #
                # use the the variable instead of arg. as a habit, we might modify the raw input.
                prams['data_file_name'] = data_file_name
            #
            if opt in ('-p', '--pickle_in'):
                if str(arg) in ('true', 'True', 'TRUE', '1'):
                    pickle_in = True
                else:
                    pickle_in=False
                prams['pickle_in'] = pickle_in
                #
            if opt in ('-q', '--pickle_out' ):
                if str(arg) in ('true', 'True', 'TRUE', '1'):
                    pickle_out = True
                else:
                    pickle_out=False
                prams['pickle_out'] = pickle_out
            #
            if opt in ('-n', '--n_cpu'):
                if arg in ('None', 'none', 'NONE', 'null', 'NULL'):
                    n_cpu=None
                else:
                    n_cpu=max(1, int(arg))
                #
                prams['n_cpu'] = n_cpu
            #
            if opt in ('-r', '--rows'):
                if arg in ('None', 'none', 'NONE', 'null', 'NULL', '-1', '0'):
                    max_rows=None
                else:
                    max_rows=int(arg)
                    if max_rows<0:
                        max_rows=None
                #
                prams['max_rows'] = max_rows
    print('*** prams: ', prams)
    #
    x = main(**prams)
