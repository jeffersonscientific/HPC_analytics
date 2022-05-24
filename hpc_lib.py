# functions, classes, and other helper bits for HPC_analytics notebooks and other libs.
#
import os
import sys
import math
import numpy
import scipy
import scipy.constants
#import matplotlib
import matplotlib.dates as mpd
#import pylab as plt
import datetime as dtm
import pytz
import multiprocessing as mpp
import pickle
import json
import h5py
#
import subprocess
import shlex
#
import numba
import pandas
#
# TODO: so... do we keep our plotting routines separate, or do we just make sure we use... uhh? (double check this)
# matplotlib.use('Agg')
#. load this first, so on an HPC primary 
import pylab as plt
#
day_2_sec=24.*3600.
#
#default_SLURM_types_dict = {'User':str, 'JobID':str, 'JobName':str, 'Partition':str, 'State':str, 'JobID_parent':str,'Timelimit':elapsed_time_2_day, 'Start':dtm_handler_default, 'Eligible':dtm_handler_default, 'Elapsed':elapsed_time_2_day, 'MaxRSS':str, 'MaxVMSize':str, 'NNodes':int, 'NCPUS':int, 'MinCPU':str, 'SystemCPU':elapsed_time_2_day, 'UserCPU':elapsed_time_2_day, 'TotalCPU':elapsed_time_2_day, 'NTasks':int}
#
# TODO: move group_ids, and local-specific things like that to site-specific  modules or data files.
#
mazama_groups_ids = ['tgp', 'sep', 'clab', 'beroza', 'lnt', 'ds', 'nsd', 'oxyvibtest', 'cardamom', 'crustal', 'stress', 'das', 'ess', 'astro', 'seaf', 'oneill', 'modules', 'wheel', 'ds2', 'shanna', 'issm', 'fs-uq-zechner', 'esitstaff', 'itstaff', 'sac-eess164', 'sac-lab', 'sac-lambin', 'sac-lobell', 'fs-bpsm', 'fs-scarp1', 'fs-erd', 'fs-sedtanks', 'fs-sedtanks-ro', 'fs-supria', 'web-rg-dekaslab', 'fs-cdfm', 'suprib', 'cees', 'suckale', 'schroeder', 'thomas', 'ere', 'smart_fields', 'temp', 'mayotte-collab']
#
# TODO: LOTS more serc_user ids!
serc_user_ids = ['biondo', 'beroza', 'sklemp', 'harrisgp', 'gorelick', 'edunham', 'sagraham', 'omramom', 'aditis2', 'oneillm', 'jcaers', 'mukerji', 'glucia', 'tchelepi', 'lou', 'segall', 'horne', 'leift']
user_exclusions = ['myoder96', 'dennis']
group_exclusions = ['modules']
#
def str2date(dt_str, verbose=0):
    try:
        return mpd.num2date(mpd.datestr2num(dt_str))
    except:
        if verbose:
            print('error date conversion: ', dt_str)
            #raise Exception('date conversion exception')
        #
        return None
#
def str2date_num(dt_str, verbose=0):
    try:
        return mpd.datestr2num(dt_str)
    except:
        if verbose:
            print('error in date-number conversion: ', dt_str)
        return None
#
def simple_date_string(dtm, delim='-'):
    # TODO: replace str() call with a proper .encode(), maybe .encode('utf-8') ?
    return delim.join([str(x) for x in [dtm.year, dtm.month, dtm.day]])
    
def datetime_to_SLURM_datestring(dtm, delim='-'):
    zf = lambda x:str(x).zfill(2)
    #
    return '{}T{}:{}:{}'.format(delim.join([str(x) for x in [dtm.year, zf(dtm.month), zf(dtm.day)]]), zf(dtm.hour), zf(dtm.minute), zf(dtm.second))
#
#@numba.jit
def elapsed_time_2_day(tm_in, verbose=0):
    #
    if tm_in in ( 'Partition_Limit', 'UNLIMITED' ) or tm_in is None:
    #if tm_in.lower() in ( 'partition_limit', 'unlimited' ):
        return None
    #
    return elapsed_time_2_sec(tm_in=tm_in, verbose=verbose)/(day_2_sec)
#
#@numba.jit
def elapsed_time_2_sec(tm_in, verbose=0):
    #
    # TODO: really??? why not just post the PL value?
    if tm_in in ( 'Partition_Limit', 'UNLIMITED' ) or tm_in is None:
    #if tm_in.lower() in ( 'partition_limit', 'unlimited' ):
        return None
    #
    days, tm = ([0,0] + list(tm_in.split('-')))[-2:]
    #
    if tm==0:
        return 0
    days=float(days)
    #
    if verbose:
        try:
            h,m,s = numpy.append(numpy.zeros(3), numpy.array(tm.split(':')).astype(float))[-3:]
        except:
            print('*** AHHH!!! error! ', tm, tm_in)
            raise Exception("broke on elapsed time.")
    else:
        h,m,s = numpy.append(numpy.zeros(3), numpy.array(tm.split(':')).astype(float))[-3:]
        
    #
    return numpy.dot([day_2_sec, 3600., 60., 1.], [float(x) for x in (days, h, m, s)])
    #return float(days)*day_2_sec + float(h)*3600. + float(m)*60. + float(s)
#
# TO_DO: can we vectorize? might have to expand/split() first, then to the math parts.
def elapsed_time_2_sec_v(tm_in, verbose=0):
    #
    tm_in = numpy.atleast_1d(tm_in)
    #
    tm_in = nm_in[numpy.logical_or(tm_in=='Partition_Limit', tm_in=='UNLIMITED')]
    # TODO: really??? why not just post the PL value?
    #if tm_in in ( 'Partition_Limit', 'UNLIMITED' ):
    #if tm_in.lower() in ( 'partition_limit', 'unlimited' ):
    #    return None
    #
    days, tm = ([0,0] + list(tm_in.split('-')))[-2:]
    #
    if tm==0:
        return 0
    days=float(days)
    #
    if verbose:
        try:
            h,m,s = numpy.append(numpy.zeros(3), numpy.array(tm.split(':')).astype(float))[-3:]
        except:
            print('*** AHHH!!! error! ', tm, tm_in)
            raise Exception("broke on elapsed time.")
    else:
        h,m,s = numpy.append(numpy.zeros(3), numpy.array(tm.split(':')).astype(float))[-3:]
        
    #
    return numpy.dot([day_2_sec, 3600., 60., 1.], [float(x) for x in (days, h, m, s)])
    #return float(days)*day_2_sec + float(h)*3600. + float(m)*60. + float(s)
#
#
def running_mean(X, n=10):
    return (numpy.cumsum(X)[n:] - numpy.cumsum(X)[:-n])/n
#
def write_sacct_batch_script():
    '''
    # write a sacct request. Since this is so fast on Sherlock, let's split this into the sacct request and the batch script. we'll probably
    #  not batch the sacct separately.
    '''
    return None
#
class SACCT_data_handler(object):
    #
    # TODO: write GRES (gpu, etc. ) handler functions.
    # TODO: add real-time options? to get sacct data in real time; we can still stash it into a file if we need to, but
    #   more likely a varriation on load_sacct_data() that executes the sacct querries, instead of loading a file.
    dtm_handler_default = str2date_num
    #
    #default_types_dict = default_SLURM_types_dict
    default_types_dict={'User':str, 'JobID':str, 'JobName':str, 'Partition':str, 'State':str, 'JobID_parent':str,
            'Timelimit':elapsed_time_2_day,
                'Start':dtm_handler_default, 'End':dtm_handler_default, 'Submit':dtm_handler_default,
                        'Eligible':dtm_handler_default,
                    'Elapsed':elapsed_time_2_day, 'MaxRSS':str,
            'MaxVMSize':str, 'NNodes':int, 'NCPUS':int, 'MinCPU':str, 'SystemCPU':elapsed_time_2_day,
                        'UserCPU':elapsed_time_2_day, 'TotalCPU':elapsed_time_2_day, 'NTasks':int}
    #
    time_units_labels={'hour':'hours', 'hours':'hours', 'hr':'hours', 'hrs':'hours', 'min':'minutes','minute':'minutes', 'minutes':'minutes', 'sec':'seconds', 'secs':'seconds', 'second':'seconds', 'seconds':'seconds'}
    #    
    time_units_vals={'days':1., 'hours':24., 'minutes':24.*60., 'seconds':24.*3600.}
    #
    #
    def __init__(self, data_file_name=None, delim='|', max_rows=None, types_dict=None, chunk_size=1000, n_cpu=None,
                 n_points_usage=1000, verbose=0, keep_raw_data=False, h5out_file=None, qs_default=[.25,.5,.75,.9], **kwargs):
        '''
        # handler object for sacct data.
        #
        #@keep_raw_data: we compute a jobs_summary[] table, which probably needs to be an HDF5 object, as per memory
        #  requirements, But for starters, let's optinally dump the raw data. we shouldn't actually need it for
        #  anything we're doing right now. if it comes to it, maybe we dump it as an HDF5 or actually build a DB of some sort.
        '''
        # TODO: reorganize a bit. Consolidate load_data() functionality into... a function, which we can rewrite
        #  for each sub-class. Then we can either similarly offload the calc_summaries() type work to a separate
        #  function or leave them in __init__() where they probably belong.
        #
        n_cpu = n_cpu or 1
        #
        if types_dict is None:
            #
            # handle all of the dates the same way. datetimes or numbers? numbers are easier for most
            #. purposes. Also, they don't morph unpredictably when transfered from one container to another.
            #. Dates are always a pain...
            #
            types_dict=self.default_types_dict
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
        #
        self.load_data()
        #
        if h5out_file is None:
            h5out_file='sacct_output.h5'
        self.h5out_file = h5out_file
        #
        self.headers = self.data.dtype.names
        # row-headers index:
        self.RH = {h:k for k,h in enumerate(self.headers)}
        self.calc_summaries()
        #
    #
    def calc_summaries(self, n_cpu=None, chunk_size=None):
        #
        n_cpu = n_cpu or self.n_cpu
        n_points_usage = self.n_points_usage
        chunk_size = chunk_size or self.chunk_size
        #
        self.jobs_summary = self.calc_jobs_summary()
        if not self.keep_raw_data:
            del self.data
        #
        #self.__dict__.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
        #
        # TODO: (re-)parallelize this...?? running continuously into problems with pickled objects being too big, so
        #   we need to be smarter about how we parallelize. Also, parallelization is just costing a lot of memory (like 15 GB/CPU -- which
        #   seems too much, so could be a mistake, but probalby not, since I'm pretty sure I ran a smilar job in SPP on <8GB).
        #   Also, on Sherlock (and presumably newer platforms in general), SPP performance is probably sufficient. I think the Mazama
        #     performance was highly IO constrained. We have other problems on Sherlock that are possibly complicated by MPP.
        
        self.cpu_usage = self.active_jobs_cpu(n_cpu=n_cpu, mpp_chunksize=min(int(len(self.jobs_summary)/n_cpu), chunk_size) )
        self.weekly_hours = self.get_cpu_hours(bin_size=7, n_points=n_points_usage, n_cpu=n_cpu)
        self.daily_hours = self.get_cpu_hours(bin_size=1, n_points=n_points_usage, n_cpu=n_cpu)
        #

    #
    def load_data(self, data_file_name=None):
        #
        data_file_name = (data_file_name or self.data_file_name)
        #
        # especially for dev, allow to pass the recarray object itself...
        if isinstance(data_file_name, str):
            # if on the off-chance we are creating hdf5, assume it is properly configured will just work:
            if os.path.splitext(data_file_name)[1] == 'h5':
                with hdf5.File(data_file_name, 'r') as fin:
                    self.data = fin['data'][:]
            else:
                self.data = self.load_sacct_data()
            #
            # If we are given a data file name as an input, and there is no hdf5 output name, create one from the data file input.
            #  otherwise, we'll use something more generic (see below).
            if h5out_file is None:
                h5out_file = '{}.h5'.format(os.path.splitext(data_file_name)[0])
        else:
            self.data = data_file_name
            #self.data = self.data_df.values.to_list()
        #
    #
    def write_hdf5(self, h5out_file=None, append=False):
        h5out_file = h5out_file or self.h5out_file
        if h5out_file is None:
            h5out_file = 'sacct_writehdf5_output.h5'
        #
        if not append:
            with h5py.File(h5out_file, 'w') as fout:
                # over-write this file
                pass
        #
        if 'data' in self.__dict__.keys():
            array_to_hdf5_dataset(input_array=self.data, dataset_name='data', output_fname=h5out_file, h5_mode='a')
        #
        array_to_hdf5_dataset(input_array=self.jobs_summary, dataset_name='jobs_summary', output_fname=h5out_file, h5_mode='a')
        array_to_hdf5_dataset(input_array=self.cpu_usage, dataset_name='cpu_usage', output_fname=h5out_file, h5_mode='a')
        array_to_hdf5_dataset(input_array=self.weekly_hours, dataset_name='weekly_hours', output_fname=h5out_file, h5_mode='a')
        array_to_hdf5_dataset(input_array=self.daily_hours, dataset_name='daily_hours', output_fname=h5out_file, h5_mode='a')
        #
        return None
    #
    def calc_jobs_summary(self, data=None, verbose=None, n_cpu=None, step_size=100000):
        '''
        # compute jobs summary from (raw)data
        # @n_cpu: number of comutational elements (CPUs or processors)
        # @step_size: numer of rows per thread (-like chunks). We'll use a Pool() (probably -- or something) to do the computation. use this
        #   variable to break up the job into n>n_cpu jobs, so we don't get a "too big to pickle" error. we'll also need to put in a little
        #  intelligence to e sure the steps are not too small, all distinct job_ids are represented, etc.
        '''
        #
        if data is None: data = self.data
        if verbose is None: verbose=self.verbose
        #
        # I think the "nonelike" syntax is actually favorable here:
        n_cpu = (n_cpu or self.n_cpu)
        n_cpu = (n_cpu or 1)
        #
        if verbose:
            print('*** calc_jobs_summary: with prams: len(data)={}, verbose={}, n_cpu={}, step_size={}'.format(len(data), verbose, n_cpu, step_size))
        #
        # NOTE: moved this out of Class to facilitate more memory efficient MPP.
        return calc_jobs_summary(data=data, verbose=verbose, n_cpu=n_cpu, step_size=step_size)
        #
    # GIT NOTE: deleted the depricated function calc_jobs_summary_depricated(self,...) which is the in-class (logic included...) version of
    #  calc_jobs_summary()
    # commit d16fd6a941c769fbce2da54e067686a1ca0b6c2f
    #
    #@numba.jit
    def load_sacct_data(self, data_file_name=None, delim=None, verbose=1, max_rows=None, chunk_size=None, n_cpu=None):
        # TODO: this should probably be a class of its own. Note that it depends on a couple of class-scope functions in
        #  ways that are not really class hierarchically compatible.
        # TODO: add capability to execute sacct queries. This is not practical on Mazama, but very feasible on Mazama.
        if data_file_name is None:
            data_file_name = self.data_file_name
        if delim is None:
            delim = self.delim
        max_rows = max_rows or self.max_rows
        chunk_size = chunk_size or self.chunk_size
        chunk_size = chunk_size or 100000
        #
        n_cpu = n_cpu or self.n_cpu
        # mpp.cpu_count() is dangerous for HPC environments. So set to 1? Maybe hedge and guess 2 or 4 (we can hyperthread a bit)?
        #n_cpu = n_cpu or mpp.cpu_count()
        n_cpu = n_cpu or 1
        #
        if verbose:
            print('** load_sact_data(), max_rows={}'.format(max_rows))
        #
        with open(data_file_name, 'r') as fin:
            headers_rw = fin.readline()
            if verbose:
                print('*** headers_rw: ', headers_rw)
            #
            headers = headers_rw[:-1].split(delim)[:-1] + ['JobID_parent']
            self.headers = headers
            #
            # make a row-handler dictionary, until we have proper indices.
            RH = {h:k for k,h in enumerate(headers)}
            self.RH = RH
            #
            if verbose:
                print('*** load data:: headers: ', headers)
            active_headers=headers
            # TODO: it would be a nice performance boost to only work through a subset of the headers...
            #active_headers = [cl for cl in headers if cl in types_dict.keys()]
            #
            # TODO: reevaluate readlines() vs for rw in...
            #
            # eventually, we might need to batch this.
            if n_cpu > 1:
                # TODO: use a context manager syntax instead of open(), close(), etc.
                P = mpp.Pool(n_cpu)
                #self.headers = headers
                #
                # TODO: how do we make this work with mpp and max_rows limit? This could be less of a problem on newer
                #  HPC systems.
                # see also: https://stackoverflow.com/questions/16542261/python-multiprocessing-pool-with-map-async
                # for the use of P.map(), using a context manager and functools.partial()
                # TODO: use an out-of-class variation of process_row() (or maybe the whole function), or modify the in-class so MPP does not need
                #   to pickle over the whole object, which breaks for large data sets. tentatively, use apply_async() and just pass
                #   headers, types_dict, and RH.
                #   def process_sacct_row(rw, delim='\t', headers=None, types_dict={}, RH={})
                # TODO: to enforce maxrows, why not just use fin.read() (and a few more small changes)?
                #  Could be that we're just running out of memory on Mazama, or we're trying to benefit from parallel file
                #  read.
                if max_rows is None:
                    results = P.map_async(self.process_row, fin, chunksize=chunk_size)
                else:
                    # This might run into memory problems, but we'll take our chances for now.
                    #X = [rw for k,rw in enumerate(fin) if k<max_rows]
                    # it's a bit faster to chunk these reads together (but probably not a major factor)
                    # Note that using readlines() will usually give > max_rows (one iteration's overage)
                    data_subset = []
                    while len(data_subset)<max_rows:
                        data_subset += fin.readlines(chunk_size)
                    #
                    results = P.map_async(self.process_row, data_subset[0:max_rows], chunksize=chunk_size)
                #
                P.close()
                P.join()
                data = results.get()
                #
                del results
                del P
            else:
                #
                # TODO: consider chunking this to improve speed (by making better use of cache?).
                data = [self.process_row(rw) for k,rw in enumerate(fin) if (max_rows is None or k<max_rows) ]
        #
        #
#            del all_the_data, rw, k, cvol, vl
        #
        if verbose:
            print('** load_sacct_data::len: ', len(data))
            self.raw_data_len = len(data)
        #
        # TODO: write a to_records() handler, so we don't need to use stupid PANDAS
        return pandas.DataFrame(data, columns=active_headers).to_records()
        #
    #@numba.jit
    def process_row(self, rw, headers=None, RH=None):
        # use this with MPP processing:
        # ... but TODO: it looks like this is 1) inefficient and 2) breaks with large data inputs because I think it pickles the entire
        #  class object... so we need to move the MPP object out of class.
        #
        headers = (headers or self.headers)
        RH      = (RH or self.RH)
        # use this for MPP processing:
        rws = rw.split(self.delim)
        #print('*** DEBUG lens: ', len(rws), len(headers))
        #return [None if vl=='' else self.types_dict.get(col,str)(vl)
        #            for k,(col,vl) in enumerate(zip(self.headers, rw.split(self.delim)[:-1]))]
        # NOTE: last entry in the row is JobID_parent.
        return [None if vl=='' else self.types_dict.get(col,str)(vl)
                    for k,(col,vl) in enumerate(zip(headers, rws[:-1]))] + [rws[RH['JobID']].split('.')[0]]
    #
    # GIT Note: Deleting the depricated get_cpu_hours_2() function here. It can be recovered from commits earlier than
    #  31 Aug 2020.
    #
    # GIT Note: Deleting depricated get_cpu_hours_depricated(). It can be restored from the current commit on 21 Sept 2020:
    # commit 44d9c5285324859d55c7878b6d0f13bbe6576be1 (HEAD -> parallel_summary, origin/parallel_summary)
    # Author: Mark Yoder <mark.yoder@gmail.com>
    #Date:   Mon Sep 21 17:29:08 2020 -0700
    def get_cpu_hours(self, n_points=10000, bin_size=7., IX=None, t_min=None, t_max=None, jobs_summary=None, verbose=False,
                     n_cpu=None, step_size=1000):
        '''
        # TODO: this in-class wrapper still needs to be tested, but since it's just a pass-through, it should work.
        #
        # a third or 4th shot at this, to get the parallelization right... again. Consider making it more procedural so we can
        #  moved the main code out of Class scope for dev.
        #  parallelize by splitting up jobs_summary and computing each subset over the full time-series, which we can compute,
        #  not pass. Since we're passing jobs_summary, maybe we don't pass the index IX any longer??? maybe we maintain that for
        # legacy.
        '''
        n_cpu = n_cpu or self.n_cpu
        n_cpu = n_cpu or 1
        if jobs_summary is None:
            jobs_summary = self.jobs_summary
        if not IX is None:
            jobs_summary = jobs_summary[IX]
        #
        return get_cpu_hours(n_points=n_points, bin_size=bin_size, t_min=t_min, t_max=t_max, jobs_summary=jobs_summary, verbose=verbose, n_cpu=n_cpu, step_size=step_size)
    #
    #@numba.jit
    def get_cpu_hours_depricated(self, n_points=10000, bin_size=7., IX=None, t_min=None, t_max=None, jobs_summary=None, verbose=False,
                     n_cpu=None):
        '''
        # Loop-Loop version of get_cpu_hours. should be more memory efficient, might actually be faster by eliminating
        #  intermediat/transient arrays.
        #
        # Get total CPU hours in bin-intervals. Note these can be running bins (which will cost us a bit computationally,
        #. but it should be manageable).
        # NOTE: By permitting jobs_summary to be passed as a param, we make it easier to do a recursive mpp operation
        #. (if n_cpu>1: {split jobs_summary into n_cpu pieces, pass back to the calling function with n_cpu=1
        '''
        #
        # stash a copy of input prams:
        inputs = {ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')}
        n_cpu = (n_cpu or self.n_cpu)
        #
        # use IX input to get a subset of the data, if desired. Note that this indroduces a mask (or something)
        #. and at lest in some cases, array columns should be treated as 2D objects, so to access element k,
        #. ARY[col][k] --> (ARY[col][0])[k]
        if jobs_summary is None:
            # NOTE: (see notes below), pasing a None index appears to have a null effect, but it actually reshapes
            #  the array, like [n,] to something like [n,1], so a vector to a [ [], [], [],...]
            jobs_summary = self.jobs_summary
        #
        if verbose:
            print('** DEBUG: len(jobs_summary): {}'.format(len(jobs_summary)))
        # NOTE: passing a None index appears to have a null effect -- just returns the whole array, but id does not.
        #  when we pass None as an index, the columns are returned with elevated rank. aka,
        #  (X[None])[col].shape == (1, n)
        #  (X[col].shape == (n,)
        #  (X[ix])[col].shape == (n,)
        if not IX is None:
            jobs_summary = jobs_summary[IX]
        #
        t_now = numpy.max([jobs_summary['Start'], jobs_summary['End']])
        #print('*** shape(t_now): ', numpy.shape(t_now))
        if verbose:
            print('** DEBUG: len(jobs_summary[ix]): {}'.format(len(jobs_summary)))
        #
        cpuh_dtype = [('time', '>f8'),
            ('t_start', '>f8'),
            ('cpu_hours', '>f8'),
            ('N_jobs', '>f8')]
        #
        if len(jobs_summary)==0:
            return numpy.array([], dtype=cpuh_dtype)
        #
        # NOTE: See above discussion RE: [None] index and array column rank.
        t_start = jobs_summary['Start']
        t_end = jobs_summary['End'].copy()
        #
        if verbose:
            print('** DEBUG: (get_cpu_hours) initial shapes:: ', t_end.shape, t_start.shape, jobs_summary['End'].shape )
        #
        # handle currently running jobs (do we need to copy() the data?)
        t_end[numpy.logical_or(t_end is None, numpy.isnan(t_end))] = t_now
        #
        if verbose:
            print('** DEBUG: (get_cpu_hours)', t_end.shape, t_start.shape)
        #
        if t_min is None:
            t_min = numpy.nanmin([t_start, t_end])
        #
        if t_max is None:
            t_max = numpy.nanmax([t_start, t_end])
        #
        # recursive MPP handler:
        #  this one is a little bit complicated by the use of linspace(a,b,n), sice linspae is inclusive for both a,b
        #  so we have to do a trick to avoid douple-calculationg (copying) the intersections (b_k-1 = a_k)
        if n_cpu>1:
            #
            # TODO: do this with a linspace().astype(int)
            time_axis = numpy.linspace(t_min, t_max, n_points)
            dk = min(int(numpy.ceil(n_points/n_cpu)),1000)
            k_ps = numpy.arange(0, n_points, dk)
            if not k_ps[-1]==n_points:
                k_ps = numpy.append(k_ps, [n_points])
            if verbose:
                print('*** k_ps: ', k_ps)
            #
            
            #
            #t_intervals = numpy.linspace(t_min, t_max, n_cpu+1)
            #dt = (t_max - t_min)/n_points
            #
            with mpp.Pool(n_cpu) as P:
                R = []
                #for k_p, (t1, t2) in enumerate(zip(t_intervals[0:-1], t_intervals[1:])):
                for k1, k2 in zip(k_ps[0:-1], k_ps[1:]):
                    t1 = time_axis[k1]
                    t2 = time_axis[k2-1]
                    
                    my_inputs = inputs.copy()
                    # TODO: need to clean up integer mismatches on n_points. I think the best thing to do is to just
                    # instantiate a full X sequence and parse it for x_min, x_max, len(x). For now, this should run.                    
                    #my_inputs.update({'t_min':t1, 't_max':t2 - dt*(k_p<float(n_cpu-1)), 'n_cpu':1, 
                    my_inputs.update({'t_min':t1, 't_max':t2, 'n_cpu':1,'n_points':int(k2-k1)})
                    if verbose:
                        print('*** my_inputs: ', my_inputs)
                    #
                    R += [P.apply_async(self.get_cpu_hours, kwds=my_inputs.copy())]
                    #
                res = [r.get() for r in R]
                # join()? not clear on this with a context manager..
                #P.join()
                #
                # create a structured array for the whole set:
                CPU_H_mpp = numpy.zeros( (0, ), dtype=cpuh_dtype)
                k0=0
                for k,r in enumerate(res):
                    #print('*** receiving r:: ', numpy.shape(r))
                    #CPU_H_mpp[k0:k0+len(r)][:] = r[:]
                    CPU_H_mpp = numpy.append(CPU_H_mpp, r)
                    k0+=len(r)
                #
                return CPU_H_mpp
        #
        #
        CPU_H = numpy.zeros( (n_points, ), dtype=cpuh_dtype)
        
        #print('*** DEBUG: shapes:: ', numpy.shape(t_now), numpy.shape(t_min), numpy.shape(t_max), numpy.shape(n_points), 
        #      numpy.shape(t_start), numpy.shape(t_end),
        #      numpy.shape(numpy.linspace(t_min, t_max, n_points)))
        #
        CPU_H['time'] = numpy.linspace(t_min, t_max, n_points)
        CPU_H['t_start'] = CPU_H['time']-bin_size
        #
        if verbose:
            print('*** ', cpu_h.shape)
        #
        for k,t in enumerate(CPU_H['time']):
            # TODO: wrap this into a (@jit compiled) function to parallelize? or add a recursive block
            #  to parralize the whole function, using an index to break it up.
            ix_k = numpy.where(numpy.logical_and(t_start<=t, t_end>(t-bin_size) ))[0]
            #print('*** shape(ix_k): {}//{}'.format(ix_k.shape, numpy.sum(ix_k)) )
            #
            N_ix = len(ix_k)
            if N_ix==0:
                CPU_H[k] = t, t-bin_size,0.,0.
                continue
            #
            #print('** ** ', ix_k)
            #CPU_H[k] = t, t-bin_size, numpy.sum( numpy.min([t*numpy.ones(N_ix), t_end[ix_k]], axis=0) -
            #                    numpy.max([(t-bin_size)*numpy.ones(N_ix), t_start[ix_k]]) )*24., N_ix
            #print('*** *** ', k,t, N_ix, ix_k)
            CPU_H[['cpu_hours', 'N_jobs']][k] = numpy.sum( (numpy.min([t*numpy.ones(N_ix), t_end[ix_k]], axis=0) -
                numpy.max( [(t-bin_size)*numpy.ones(N_ix), t_start[ix_k]], axis=0))*24.*(jobs_summary['NCPUS'])[ix_k] ), N_ix
        #
        #CPU_H['cpu_hours']*=24.
        #print('*** returning CPU_H:: ', numpy.shape(CPU_H))
        return CPU_H
    #
    def active_jobs_cpu(self, n_points=5000, ix=None, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, jobs_summary=None, verbose=None, mpp_chunksize=None):
        '''
        ##
        # Use out-of-class procedural function to improve parallelization (or at least development of parallel code).
        #  this class function will just be a wrapper and will retain some of the default variable handlers.
        #
        # @n_points: number of points in returned time series.
        # @bin_size: size of bins. This will override n_points
        # @t_min: start time (aka, bin phase).
        # @ix: an index, aka user=my_user
        '''
        if verbose is None:
            verbose = self.verbose
        #
        mpp_chunksize = mpp_chunksize or self.chunk_size
        n_cpu = n_cpu or self.n_cpu
        #
        if (not ix is None) and len(ix)==0:
            #return numpy.array([(0.),(0.),(0.)], dtype=[('time', '>f8'),
            #return numpy.array([], dtype=[('time', '>f8'),('N_jobs', '>f8'),('N_cpu', '>f8')])
            return null_return()
        #
        if jobs_summary is None:
            jobs_summary=self.jobs_summary
        if not ix is None:
            jobs_summary=jobs_summary[ix]
        #
        return active_jobs_cpu(n_points=n_points, bin_size=bin_size, t_min=t_min, t_max=t_max, t_now=t_now, n_cpu=n_cpu, jobs_summary=jobs_summary,
            verbose=verbose, mpp_chunksize=mpp_chunksize)
    #
    # GIT Note (commit d16fd6a941c769fbce2da54e067686a1ca0b6c2f): Removing active_jobs_cpu_DEPRICATED(self,...).
    #   functional code moved out of class to facilitate development (and MPP mabye??)
    #
    # NOTE: some trial and error figuring out how to offload process_ajc_row() to facilitate parallelization.
    #  Ultimately, MPP gains were significant but costly (did not generally parallelize well), and a PAIN to maintain.
    #  on Sherlock, we seem to have sufficient performance to process SPP, so some of this can mabye be simplified?
    # GIT note (commit d16fd6a941c769fbce2da54e067686a1ca0b6c2f):  removing process_ajc_row_2(), as per deprication.
    #@numba.jit
    #def process_ajc_row(self,j, t_start, t_end, t, jobs_summary):
    def process_ajc_row(self, t):
        '''
        # Active Jobs CPU worker function...
        '''
        # DEPRICATION: moving this outside the class definition to better facilitate MPP and development.
        #print('*** DEBUG: t:: {}'.format(t) )
        ix_t = numpy.where(numpy.logical_and(self.t_start<=t, self.t_end>t))[0]
        #output[['N_jobs', 'N_cpu']][j] = len(ix_t), numpy.sum(jobs_summary['NCPUS'][ix_t])
        return len(ix_t), numpy.sum(self.ajc_js['NCPUS'][ix_t])
        #
    def get_wait_stats(self, qs=None):
        # TODO: revise this to use a structured array, not a recarray.
        dtype=[('ncpus', '>i8'), ('mean', '>f8'), ('median', '>f8'),  ('stdev', '>f8'), ('min', '>f8'),  ('max', '>f8')]
        if not qs is None:
            for k,q in enumerate(qs):
                dtype += [(f'q{k+1}', '>f8')]
        wait_stats = numpy.core.records.fromarrays(numpy.zeros((len(self.jobs_summary), len(dtype))).T,
                                                   dtype=dtype)
        wait_stats.qs=numpy.array(qs)
        #
        # NOTE: jobs that have not started will be blank? None? NaN? Let's see if we can't just pick up NaNs on the flipside.
        delta_ts = self.jobs_summary['Start'] - self.jobs_summary['Submit']
        ix_started = numpy.invert(numpy.isnan(delta_ts))
        delta_ts = delta_ts[ix_started]
        #
        for j,k in enumerate(sorted(numpy.unique(self.jobs_summary['NCPUS']) ) ):
            #
            x_prime = delta_ts[numpy.logical_and(self.jobs_summary['NCPUS'][ix_started]==k, delta_ts>=0.)]
            #wait_stats[k-1]=[[k, numpy.mean(x_prime), numpy.median(x_prime), numpy.std(x_prime), 
            #                 numpy.min(x_prime), numpy.max(x_prime)]]
            #
            #wait_stats[k-1][0] = k
            wait_stats[j][0] = k
            if len(x_prime)==0:
                continue

            for l,f in zip(range(1, 6), [numpy.mean, numpy.median, numpy.std, numpy.min, numpy.max]):
                #
                wait_stats[j][l]=f(x_prime)
            #
            # I NEVER remember the right syntax to set subsets of recarreays or structured arrays...
            #quantiles = numpy.quantiles(x_prime, qs)
            #print('*** DEBUG qs: {}, nanlen: {}'.format(qs,numpy.sum(numpy.isnan(x_prime))))
            #
            if not qs is None:
              for k, (q_s, q_v) in enumerate(zip(qs, numpy.quantile(x_prime, qs))):
                  wait_stats[[f'q{k+1}']][j] = q_v
        #
        return wait_stats
    #
    def submit_wait_distribution(self, n_cpu=1):
        # TODO: add input index?
        #
        n_cpu = numpu.atleast_1d(n_cpu)
        #
        X = self.jobs_summary['Start'] - self.jobs_summary['Submit']
        #
        return numpy.array([n_cpu, [X[self.jobs_summary['n_cpu']==n] for n in n_cpu]]).T
    #
    def get_wait_times_per_ncpu(self, n_cpu):
        # get wait-times for n_cpu cpus. to be used independently or as an MPP worker function.
        #
        ix = self.jobs_summary['NCPUS']==n
        return self.jobs_summary['Start'][ix] - self.jobs_summary['Submit'][ix]
    #
    def export_primary_data(self, output_path='data/SACCT_full_data.csv', delim='\t'):
        # DEPRICATION: This does not appear to work very well, and we have HDF5 options, so consider this depricated
        #  and maybe remove it going forward.
        #
        # NOTE: we may be be deleting the primary data, since it is really not useful (the jobs_summary is what we really want),
        #  so this function could be (mostly) broken at this point.
        #
        #pth = pathlib.Path(outputy_path)
        #
        # in the end, this is quick and easy, but for a reason -- it does not work very well.
        #. same goes for the quick and easy PANDAS file read method. it casts strings and other
        #. unknowns into Object() classes, which we basically can't do anything with. to reliably
        #. dump/load data to/from file, we need to handle strings, etc. but maybe we just pickle()?
        # ... or see hdf5 options now mostly developed.
        return self.data.tofile(outputy_path,
                                        format=('%s{}'.format(delim)).join(self.headers), sep=delim)
    #
    def export_summary_data(self, output_path='data/SACCT_summaruy_data.csv', delim='\n'):
        # DEPRICATION: This does not appear to work very well, and we have HDF5 options, so consider this depricated
        #  and maybe remove it going forward.
        #
        return self.jobs_summary.tofile(output_path,
                                        format='%s{}'.format(delim).join(self.headers), sep=delim)
    #
    def get_run_times(self, include_running_jobs=True, IX=None):
        '''
        # NOTE: this should be equivalent to the Elapsed colume, as automagicallhy computed by SLURM.
        # compute job run times, presumably to compute a distribution.
        # @include_running_jobs: if True, then set running jobs (End time = nan --> End time = now()
        # @IX: optional index to subset the data
        '''
        #
        starts = (self.jobs_summary['Start'])[IX].copy()
        ends   = (self.jobs_summary['End'])[IX].copy()
        if include_running_jobs:
            ends[numpy.isnan(ends)]=mpd.date2num(dtm.datetime.now())
        #
        ix = numpy.logical_and(numpy.invert(numpy.isnan(starts) ), numpy.invert(numpy.isnan(ends) ) )
        #
        return ends[ix] - starts[ix]
    #
    def get_submit_compute_vol_timeofday(self, time_units='hours', qs=[.5, .75, .95], time_col='Submit', timelimit_default=None):
        '''
        # wait times as a functino fo time-of-day. Compute various quantiles for these values.
        # @time_units: {hours, days, and somme others}. References class scope time_units_labels consolidation dict and
        #.   self.time_units_vals{} to get the values for those. Base units are days, so ('hour', 'hours' 'hr') -> 24 (hrs/day)
        # @qs=[]: quantiles. default, [.5, .75, .95] give s the .5, .75, .95 quantiles.
        # @time_col: jobs_summary time column. Available columns include ('Submit', 'Start', 'End', 'Eligible'). Will
        #. default to 'Submit'
        #
        # see also get_submit_wait_timeofday()
        '''
        #
        # NOTE: specifying columns...
        # , Y_cols=['NCPUS', 'Timelimit'])
        #. it's pretty easy, in principle, to set Y_cols as a user defined parameter, except that each column is likely
        #. to have its own None/NaN special needs handling requirements. One solution would be to write the more generalized
        #. handler, and then wrapepr functions to prepare the input arrays.
        #
        if timelimit_default is None:
            timelimit_default = numpy.median((self.jobs_summary['Timelimit'])[numpy.invert(numpy.isnan(self.jobs_summary['Timelimit']))])
        #
        if (isinstance(time_units, int) or isinstance(time_units, float) ):
            u_time = time_units
        else:
            time_units = self.time_units_labels.get(time_units,'hours')
            u_time = self.time_units_vals.get(time_units, 24)
        #
        self.get_submit_compute_vol_timeofday_prams=locals().copy()
        #
        # do we have a valid time_col? for now, require an exact match. 
        if not time_col in self.jobs_summary.dtype.names:
            time_col='Submit'
        #
        # get a nan-free index or jobs_summary subset
        JS=self.jobs_summary[numpy.invert(numpy.isnan(self.jobs_summary[time_col]))]
        #
        #tf = time_units_dict[time_units]
        #
        #X = ( (self.jobs_summary[time_col]*u_time)%u_time).astype(int)
        X = ( (JS[time_col]*u_time)%u_time).astype(int)
        
        X0 = numpy.unique(X)
        #
        #timelimit = self.jobs_summary['Timelimit'].copy()
        timelimit = JS['Timelimit'].copy()
        timelimit[numpy.isnan(timelimit)]=timelimit_default
        #
        #Y = self.jobs_summary['NCPUS']*timelimit
        Y = JS['NCPUS']*timelimit
        
        # this generalized syntax would be nice, except that we need custom NaN handline...
        # this syntax should work, but sometimes seems to get confused by masks, reference, etc. So
        #.  maybe better to just to a little bit of Python LC.
        #Y = numpy.prod(self.jobs_summary[Y_cols], axis=1)
        #Y = numpy.prod([self.jobs_summary[cl] for cl in Y_cols], axis=0)
        #
        # this could be expedited by sorting, then using numpy.searchsorted(), but this syntax is so easy
        #. and for now, it's fast enough.
        #quantiles = numpy.array([numpy.quantile(Y[X==j], qs) for j in X0])
        #means = numpy.array([numpy.mean(Y[X==j]) for j in X0])
        q_sum = numpy.array([numpy.sum(Y[X==j]) for j in X0])
        #cpu_sum = numpy.array([numpy.sum(self.jobs_summary['NCPUS'][X==j]) for j in X0])
        cpu_sum = numpy.array([numpy.sum(JS['NCPUS'][X==j]) for j in X0])
        #
        # wrap up in an array:
        #dts =[('time', '>f8'), ('mean', '>f8')] + [('q{}'.format(q), '>f8') for q in qs]
        #    
        #    
        return numpy.core.records.fromarrays([X0, q_sum, cpu_sum],
                                      dtype=[('time', '>f8'), ('cpu-time', '>f8'), ('cpus', '>f8')] )
        #                                     + [('q{}'.format(k), '>f8') for k,q in enumerate(qs)] )
    #
    def get_submit_wait_timeofday(self, time_units='hours', qs=numpy.array([.5, .75, .95])):
        '''
        # wait times as a function fo time-of-day. Compute various quantiles for these values.
        # @time_units: {hours, days, and somme others}. References class scope time_units_labels consolidation dict and
        #.   self.time_units_vals{} to get the values for those. Base units are days, so ('hour', 'hours' 'hr') -> 24 (hrs/day)
        # @qs=[]: quantiles. default, [.5, .75, .95] give s the .5, .75, .95 quantiles.
        '''
        #
        # TODO: allow units and modulus choices.
        
        #
        # allow a custom, numerical value to be passed for time_units (don't know why you would do this, but...)
        if (isinstance(time_units, int) or isinstance(time_units, float) ):
            u_time = time_units
        else:
            time_units = self.time_units_labels.get(time_units,'hours')
            u_time = self.time_units_vals.get(time_units, 24.)
        #
        #tf = time_units_dict[time_units]
        #
        #X = (self.jobs_summary['Submit']*24.)%24
        X = (self.jobs_summary['Submit']*u_time)%u_time
        Y = (self.jobs_summary['Start'] - self.jobs_summary['Submit'] )*u_time
        #
        # TODO: we were sorting for a reason, but do we still need to?
        #  NOTE: also, we're sorting on the modulo sequence, so it's highly ambigous. My guess is we can just get rid of this,\
        # but it's worth testing first.
        ix = numpy.argsort(X)
        X = X[ix].astype(int)    # NOTE: at some point, we might actually want the non-int X values...
        Y = Y[ix]
        X0 = numpy.unique(X).astype(int)
        #
        # NOTE: qualtile is a relatively new numpuy feature (1.16.x I think). It's not uncommon
        #  to run into 1.14 or 1.15, so this breaks. For now, let's use .percen() instead,
        #  which is equivalnet except that q[0,100] instead of q[0,1]
        # TODO: implement percentile()
        quantiles = numpy.array([numpy.percentile(Y[X==j], 100*numpy.array(qs)) for j in X0])
        #quantiles = numpy.array([numpy.quantile(Y[X==j], qs) for j in X0])
        #
        #quantiles = numpy.array([numpy.quantile(y[X.astype(int)==int(j)], qs) for j in range(24)])
        #
        # let's do a quick check to see if these numbers are being computed correctly. so
        #. let's do in the hard way 
        # These numbers apepar to check out, so let's keep the mean() calc, but do it a faster way.
        #means = numpy.array([numpy.mean([y for x,y in zip(X,Y) if int(x)==j]) for j in X0])
        #medians = numpy.array([numpy.median([y for x,y in zip(X,Y) if int(x)==j]) for j in X0])
        means = numpy.array([numpy.mean(Y[X==j]) for j in X0])
        #
        # wrap up in an array:
        #dts =[('time', '>f8'), ('mean', '>f8')] + [('q{}'.format(q), '>f8') for q in qs]
            
            
        return numpy.core.records.fromarrays(numpy.append([X0, means], quantiles.T, axis=0),
                                              dtype=[('time', '>f8'), ('mean', '>f8')] +
                                             [('q{}'.format(k), '>f8') for k,q in enumerate(qs)] )
    #
    #
    # figures and reports:
    def active_cpu_jobs_per_day_hour_report(self, qs=[.45, .5, .55], figsize=(14,10),
     cpu_usage=None, verbose=0, foutname=None, periodic_projection='rectilinear'):
        '''
        # 2x3 figure of instantaneous usage (active cpus, jobs per day, week)
        '''
        if cpu_usage is None:
            cpu_usage = self.cpu_usage
        #
        # Polar, etc. projections:
        # allowed projections (probably) include:
        #  {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}
        #  default, None -> 'rectilinear'
        fg = plt.figure(figsize=figsize)
        #
        ax1 = fg.add_subplot(2,3,1)
        ax2 = fg.add_subplot(2,3,2, projection=periodic_projection)
        ax3 = fg.add_subplot(2,3,3, projection=periodic_projection)
        ax4 = fg.add_subplot(2,3,4)
        ax5 = fg.add_subplot(2,3,5, projection=periodic_projection)
        ax6 = fg.add_subplot(2,3,6, projection=periodic_projection)
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        #
        if periodic_projection == 'polar':
            daily_Period = 2.0*scipy.constants.pi/24.
            weekly_Period = 2.0*scipy.constants.pi/7.
            #
            for ax in (ax2, ax3, ax5, ax6):
              ax.set_theta_direction(-1)
              ax.set_theta_offset(math.pi/2.0)
        else:
            daily_Period = 1.0
            weekly_Period = 1.0
        #
        #qs = [.45, .5, .55]
        qs_s = ['q_{}'.format(q) for q in qs]
        #print('*** qs_s: ', qs_s)
        #
        cpu_hourly = time_bin_aggregates(XY=numpy.array([cpu_usage['time'],
                                                         cpu_usage['N_cpu']]).T, qs=qs)
        jobs_hourly = time_bin_aggregates(XY=numpy.array([cpu_usage['time'],
                                                          cpu_usage['N_jobs']]).T, qs=qs)
        #
        # TODO: poorly handling the DoW here. For some reason (that made sense at the time -- maybe to do with needing an integer),
        #  we do a little trick here where we divide/multiply the input sequence and then always use %1 (x%7 == ((x/7)%1)*7 )... or something.
        #  so
        cpu_weekly = time_bin_aggregates(XY=numpy.array([cpu_usage['time']/7.,
                                                         cpu_usage['N_cpu']]).T, bin_mod=7., qs=qs)
        jobs_weekly = time_bin_aggregates(XY=numpy.array([cpu_usage['time']/7.,
                                                          cpu_usage['N_jobs']]).T, bin_mod=7., qs=qs)
        cpu_weekly['x'] = (cpu_weekly['x']+3)%7
        jobs_weekly['x'] = (jobs_weekly['x']+3)%7
        cpu_weekly.sort(order='x')
        jobs_weekly.sort(order='x')
        #
        # correct for UTC/PST? Still not positive this is correct:
        # Also... I think this index is sort of a dumb way to shift the time. It would be better to just sort the array...
        #  maybe not if you want to keep both sequences though.
        # That said, looks like SLURM uses local time. For now, just get rid of the offset.
        #ix_pst = numpy.argsort( (jobs_hourly['x']-7)%24)
        # TODO: this *is* a sort of dumb (certainly confusing...) way to re-sort the Y-axis to relabel the x-axis. since SLURM
        #  appears to use PST, we should just get rid of ix_pst.
        ix_pst = numpy.argsort( (jobs_hourly['x']-0)%24)
        #
        X_tmp_hr = numpy.linspace(jobs_hourly['x'][0], jobs_hourly['x'][-1]+1,200)*daily_Period
        X_tmp_wk = numpy.linspace(jobs_weekly['x'][0], jobs_weekly['x'][-1]+1,200)*weekly_Period
        #
        hh1 = ax1.hist(sorted(cpu_usage['N_jobs'])[0:int(1.0*len(cpu_usage))], bins=25, cumulative=False)
        #ax2.plot(jobs_hourly['x'], jobs_hourly['mean'], ls='-', marker='o', label='PST')
        ln, = ax2.plot(jobs_hourly['x']*daily_Period, jobs_hourly['mean'][ix_pst], ls='-', marker='o', label='Local Time')
        clr = ln.get_color()
        #
        #ax2.plot(jobs_hourly['x']*daily_Period, numpy.ones(len(jobs_hourly['x']))*numpy.mean(jobs_hourly['mean'][ix_pst]), ls='--', marker='', zorder=1, alpha=.7, label='mean', color=clr)
        ax2.plot(X_tmp_hr, numpy.ones(len(X_tmp_hr))*numpy.mean(jobs_hourly['mean'][ix_pst]), ls='--', marker='', zorder=1, alpha=.7, label='mean', color=clr)
        #
        ax3.plot(jobs_weekly['x']*weekly_Period, jobs_weekly['q_0.5'], ls='-', marker='o', color=clr)
        #ax3.plot(jobs_weekly['x']*weekly_Period, jobs_weekly['mean'], ls='--', marker='', color=clr, label='mean')
        ax3.fill_between(jobs_weekly['x'], jobs_weekly[qs_s[0]], jobs_weekly[qs_s[-1]],
                         alpha=.1, zorder=1, color=clr)
        ax3.plot(X_tmp_wk, numpy.ones(len(X_tmp_wk))*numpy.mean(jobs_weekly['q_0.5']), ls='-.', alpha=.7, zorder=1, label='$<q_{0.5}$')
        #
        #
        hh4 = ax4.hist(cpu_usage['N_cpu'], bins=25)
        #ax5.plot(cpu_hourly['x'], cpu_hourly['mean'], ls='-', marker='o', label='PST')
        ln, = ax5.plot( cpu_hourly['x']*daily_Period, cpu_hourly['mean'][ix_pst], ls='-', marker='o', label='Local Time')
        clr=ln.get_color()
        ax5.plot(X_tmp_hr, numpy.ones(len(X_tmp_hr))*numpy.mean(cpu_hourly['mean'][ix_pst]), ls='--', marker='',
                color=clr, label='mean')
        #
        ax6.plot(cpu_weekly['x']*weekly_Period, cpu_weekly['q_0.5'], ls='-', marker='o', color=clr)
        ax6.plot(cpu_weekly['x']*weekly_Period, cpu_weekly['mean'], ls='--', marker='', color=clr, label='mean')
        #
        # TODO: can we simplyfy this qs syntax?
        ax6.fill_between(cpu_weekly['x'], cpu_weekly[qs_s[0]], cpu_weekly[qs_s[-1]], alpha=.1, zorder=1, color='b')
        ax6.plot(X_tmp_wk, numpy.ones(len(X_tmp_wk))*numpy.mean(cpu_weekly['q_0.5']), ls='-.', alpha=.7, zorder=1, label='$<q_{0.5}$')
        #
        #
        #ax1.set_ylim(-5., 200)
        ax1.set_title('$N_{jobs}$ Histogrm', size=16)
        ax2.set_title('Hour-of-day job counts', size=16)
        ax3.set_title('Day-of-week job counts', size=16)
        #
        ax4.set_title('$N_{cpu}$ Histogram', size=16)
        ax5.set_title('Hour-of-day CPU counts', size=16)
        ax6.set_title('Day-of-week CPU counts', size=16)
        #
        ax5.set_xlabel('Hour of Day (PST)')
        plt.suptitle('Instantaneous Usage ', size=16)
        #
        ax2.legend(loc=0)
        ax5.legend(loc=0)
        #
        weekdays={ky:vl for ky,vl in zip(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])}
        for ax in (ax3, ax6):
            #
            #print('*** wkly', weekly_Period*numpy.array(ax.get_xticks()))
            #ax.set_xticklabels(['', *[weekdays[int(x/weekly_Period)] for x in ax.get_xticks()[1:]] ] )
            #ax.set_xticks(numpy.arange(0,8)*weekly_Period)
            
            ax.set_xticks(cpu_weekly['x']*weekly_Period)
            ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][0:len(cpu_weekly['x'])])
            
            #ax.set_xticklabels(['', 'sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
            #ax.grid()
        for ax in (ax2,ax5):
            #
            #print('*** ', ax.get_xticks() )
            tmp_x = ax.set_xticklabels(['', *[str(int((x%24)/daily_Period) ) for x in ax.get_xticks()[1:]]])
            #ax.grid()
        #
        if not foutname is None:
            pth,nm = os.path.split(foutname)
            if not os.path.isdir(pth):
                os.makedirs(pth)
            #
            plt.savefig(foutname)
        #
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.grid(True)
        #
        return {'cpu_hourly':cpu_hourly, 'jobs_hourly':jobs_hourly, 'cpu_weekly':cpu_weekly, 'jobs_weekly':jobs_weekly}
    #
    def compute_vol_distributions(self, x_col='NNodes', normed=True):
        '''
        # PDF and CDF of compute-volume (V = ncpus*elapsed_time
        # @x_col: x_col, so distribution as a function of x_col
        # @normed: normalize pdf, so sum(pdf)=1; cdf[-1] = 1
        '''
        #
        # shorthand reference:
        X = self.jobs_summary
        #
        compute_volume = X['NCPUS']*X['Elapsed']
        #
        N1,N2 = numpy.unique(X[x_col])[numpy.array([0,-1])]
        #
        # given the simplicty of continuous, sequential indexing, use an array (not a dict), and there seems to be
        #  some value in using a structured array.
        cv_dist = numpy.zeros(shape=(N2-N1+1,), dtype=[(x_col, '>i8'), ('pdf', '>f8'), ('cdf', '>f8')]  )
        cv_dist[x_col] = numpy.arange(N1,N2+1)
        #
        for n, v in zip(X[x_col], compute_volume):
            cv_dist['pdf'][n-1] += v
        #
        if normed:
            cv_dist['pdf']/=numpy.sum(cv_dist['pdf'])
        cv_dist['cdf']=numpy.cumsum(cv_dist['pdf'])
        #
        return cv_dist
    #
    def compute_vol_distributions_report(self, x_col='NNodes', normed=True,
                                     figsize=None, ax1=None, ax2=None):
        #
        distributions = self.compute_vol_distributions(x_col=x_col, normed=normed)
        #
        if figsize is None:
            figsize=(12,10)
        if ax1 is None or ax2 is None:
            # valid axes not provided:
            fg = plt.figure(figsize=figsize)
            #
            ax1 = fg.add_subplot('211')
            ax2 = fg.add_subplot('212', sharex=ax1)
            ax1.grid()
            ax2.grid()
        else:
            fg = ax1.figure
        #
        x     = distributions[x_col]
        y     = distributions['pdf']
        y_cum = distributions['cdf']
        #
        ax1.plot(x,y, ls='-', marker='o', lw=3)
        #
        ax2.plot(x,y_cum, ls='-', marker='o', lw=3)
        #
        ax1.fill_between(x[0:5], (y)[0:5], numpy.ones(5)*0., alpha=.2)
        ax2.fill_between(x[0:5], (y_cum)[0:5], numpy.ones(5)*y_cum[0], alpha=.2)
        #
        ax1.set_xlim(-1, 30)
        #
        fg.suptitle('Compute Volume $(N_{cpu} \cdot \Delta t)$', size=16)
        ax1.set_ylabel('PDF', size=16)
        ax2.set_ylabel('CDF', size=16)
        ax2.set_xlabel('Nodes ($N_{nodes}$)', size=16)

        print('nodes, pdf, cdf')
        for k, (n, p1, p2) in enumerate(zip(x, y, y_cum)):
            print('{}: {}, {}'.format(n,p1,p2))
            #
            if k>10:break
        #
        return distributions
    #

class SACCT_data_direct(SACCT_data_handler):
    format_list_default = ['User', 'Group', 'GID', 'Jobname', 'JobID', 'JobIDRaw', 'partition', 'state', 'time', 'ncpus',
               'nnodes', 'Submit', 'Eligible', 'start', 'end', 'elapsed', 'SystemCPU', 'UserCPU',
               'TotalCPU', 'NTasks', 'CPUTimeRaw', 'Suspended', 'ReqTRES', 'AllocTRES']
    def __init__(self, group=None, partition=None, delim='|', start_date=None, end_date=None, more_options=[], delta_t_days=10, default_catalog_length=180,
        format_list=None, n_cpu=None, types_dict=None, verbose=0, chunk_size=1000,
        h5out_file=None, keep_raw_data=False, raw_output_file=None, n_points_usage=1000,
        **kwargs):
        #
        # TODO: this is a mess. I think it needs to be a bit nominally sloppier, then cleaner -- so maybe read the first block,
        #  characterize the data, then process the rest of the elements, or read the data (in parallel), then process in parallel.
        #  rather than trying to do it all in one call...
        #  TODO: wrap all the data loading into a data_load() function, which will need to be nullified to not inherit\
        #   from the parent class.
        #
        #
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})
        options = [tuple(rw) if len(rw)>=2 else (rw,None) for rw in more_options]
        options += [tuple((ky[6:],vl)) for ky,vl in kwargs.items() if ky.lower().startswith('sacct_')]
        sacct_str_template = 'sacct {} -p --allusers --starttime={} --endtime={} --format={} '
        #
        if types_dict is None:
            types_dict=self.default_types_dict
        #
        format_list = format_list or self.format_list_default
        format_string = ','.join(format_list)
        #
        n_cpu = n_cpu or min(6, mpp.cpu_count() )
        #
        if not group is None:
            options += [('group', group)]
        if not partition is None:
            options += [('partition', partition)]
        #
        try:
            delta_t_days=int(delta_t_days)
        except:
            delta_t_days=30
            if verbose:
                print('** Warning: delta_t_days input not valid. set to default value={}'.format(delta_t_days))
            #
        #
        options += [('delimiter', '"{}"'.format(delim))]
        #
        options_str=''
        for (op,vl) in options:
            if vl is None:
                options_str += f'-{op}'
            else:
                if len(op)==1:
                    options_str += f' -{op} {vl} '
                elif len(op)>1:
                    options_str += f' --{op}={vl} '
            #
        #
        ######################
        # process start_/end_date
        if end_date is None or end_date == '':
            end_date = dtm.datetime.now()
        if start_date is None or start_date == '':
            start_date = end_date - dtm.timedelta(days=default_catalog_length)
        #
        if isinstance(start_date, str):
            start_date = str2date(start_date)
        if isinstance(end_date, str):
            end_date = str2date(end_date)
        start_date = start_date.replace(tzinfo=pytz.UTC)
        end_date   = end_date.replace(tzinfo=pytz.UTC)
        #
        print('*** ', start_date, type(start_date), end_date, type(end_date))
        # end process start_/end_date
        ###############
        #
        start_end_times = [(start_date, min(start_date + dtm.timedelta(days=delta_t_days), end_date))]
        while start_end_times[-1][1] < end_date:
            start_end_times += [[start_end_times[-1][1], min(start_end_times[-1][1] + dtm.timedelta(days=delta_t_days), end_date) ]]
        #
        self.start_end_times = start_end_times
        ##########################
        #
        self.__dict__.update({ky:val for ky,val in locals().items() if not ky in ('self', '__class__')})
        #
        print('*** DEBUG: Now execute load_sacct_data(); options_str={}'.format(options_str))
        self.data = self.load_sacct_data(options_str=options_str, format_string=format_string)
        #
        if not raw_output_file is None:
            if raw_output_file.endswith('h5'):
                # TODO: Not totally positive this works...
                dta_to_h5 = array_to_hdf5_dataset(input_array=self.data, dataset_name='raw_data', output_fname=raw_output_file, h5_mode='w', verbose=0)
            else:
              with open('{}.csv'.format(os.path.splitext(raw_output_file)[0]), 'w' ) as fout:
                  #fout.write('\t'.join(self.headers))
                  fout.write('{}\n'.format('\t'.join(self.data.dtype.names)) )
                  for rw in self.data:
                      #fout.write('\t'.join(rw[:-1].split(delim)))
                      #fout.write('{}\n'.format(rw))
                      fout.write('{}\n'.format('\t'.join(list(rw))))
              #
        #
        print('*** DEBUG: load_sacct_data() executed. Compute calc_jobs_summary()')
        print('*** DEBUG: data stuff: ', len(self.data), self.data.dtype)
        #self.jobs_summary=self.calc_jobs_summary(data)
        #
        super(SACCT_data_direct, self).__init__(delim=delim, start_date=start_date, end_date=end_date, h5out_file=h5out_file, keep_raw_data=keep_raw_data,n_points_usage=n_points_usage, n_cpu=n_cpu, types_dict=types_dict, verbose=verbose, chunk_size=chunk_size, **kwargs)
        #
        #del data
        self.__dict__.update({ky:val for ky,val in locals().items() if not ky in ('self', '__class__')})
    #
    def load_data(self, *args, **kwargs):
        pass
    #
    def load_sacct_data(self, options_str='', format_string='', n_cpu=None, start_end_times=None, max_rows=None, verbose=None):
        #
        n_cpu = n_cpu or self.n_cpu
        n_cpu = n_cpu or 4
        #
        # might want to force n_cpu>1...
        #n_cpu = numpy.min(n_cpu,2)
        #
        #, raw_data_out_file=None
        #raw_data_out_file = raw_data_out or self.raw_data_out_file
        #
        verbose = verbose or self.verbose
        verbose = verbose or False
        #
        start_end_times = start_end_times or self.start_end_times
        sacct_str_template = self.sacct_str_template
        #
        # for reference, stash a copy of the ideal sacct str (ie, with full date range)
        self.sacct_str = sacct_str_template.format(options_str, datetime_to_SLURM_datestring(start_end_times[0][0]),
                                    datetime_to_SLURM_datestring(start_end_times[-1][-1]), format_string )
        if verbose:
          print(f'** DEBUG: sacct_str:: {self.sacct_str}')
        #
        if n_cpu == 1:
            #sacct_out = None
            data = None
            for k, (start, stop) in enumerate(start_end_times):
                if verbose:
                    print('** DEBUG: processing SACCT data, n_cpu=1, start: {}, end: {}'.format(start, stop))
                #
                #sacct_str = 'srun sacct {} {} -p --allusers --starttime={} --endtime={} --format={} '.format( ('--noheader' if k>0 else ''),
                #                        options_str, datetime_to_SLURM_datestring(start),
                #                        datetime_to_SLURM_datestring(stop), format_string )
                sacct_str = sacct_str_template.format(options_str, datetime_to_SLURM_datestring(start),
                                    datetime_to_SLURM_datestring(stop), format_string )
                #
                if verbose: print('** [{}]: {}\n'.format(k, sacct_str))
                #
                #sacct_out += subprocess.run(sacct_str.split(), stdout=subprocess.PIPE).stdout.decode()
                # FIXME:
                #TODO: this is not appending correctly. Maybe return as lists and construct structued array here?
                #if sacct_out is None:
                if data is None:
                    #sacct_out = self.get_and_process_sacct_data(sacct_str)
                    XX = self.get_and_process_sacct_data(sacct_str, with_headers=True, as_dict=True)
                    data = XX['data']
                    headers = XX['headers']
                else:
                    # will appending work, or do we need to handle the headers, etc. in a more sophisticated way? This is still
                    #  inefficient; there is a bit of copying, etc. that should be optimized later.
                    # TODO: No. Appending does not work; something does not promote (or something) needs to be returned as a list or
                    #  collected and copied into the array like in the MPP model.
                    #
                    #sacct_out = numpy.append(sacct_out, self.get_and_process_sacct_data(sacct_str) )
                    data += self.get_and_process_sacct_data(sacct_str, with_headers=False)
                #
            #
            sacct_out = pandas.DataFrame(data, columns=headers).to_records()
            #
            del data
        else:
            # TODO: consolidate spp,mpp methods...
            with mpp.Pool(n_cpu) as P:
                R = []
                for k, (start, stop) in enumerate(start_end_times):
                    sacct_str = sacct_str_template.format(
                        options_str, datetime_to_SLURM_datestring(start),
                        datetime_to_SLURM_datestring(stop), format_string )
                    #
                    kw_prams = {'sacct_str':sacct_str, 'delim':self.delim, 'with_headers':True,'as_dict':False }
                    if not max_rows is None:
                        kw_prams['max_rows'] = int(numpy.ceil(max_rows/n_cpu))
                    #
                    if verbose: print('*** DEBUG: sacct_str: ', sacct_str )
                    R += [P.apply_async(self.get_and_process_sacct_data, kwds=kw_prams )]
                #
                # is this the right syntax?
                Rs = [r.get() for r in R]
            #
            #sacct_out = Rs[0]
            len_total = numpy.sum([len(x) for x in Rs])
            sacct_out = numpy.zeros((len_total, ), dtype=Rs[0].dtype)
            self.headers = sacct_out.dtype.names
            k0=0
            for k,R in enumerate(Rs):
                n = len(R)
                #
                if verbose: print('*** DEBUG: len(R): ', len(R) )
                #
                for col in R.dtype.names:
                    sacct_out[col][k0:k0+n][:]=R[col]
                #
                k0+=n
            #
        #
        #if not raw_data_out_file is None:
        # 'index' column will be unique only to each process, so reset to the whole set. We use it to make sorting
        #  unambiguous. Note that breaking up the sacct query creates lots of duplicate records. When we also have
        #  duplicat 'index', we get NoneType sorting errors.
        sacct_out['index'] = numpy.arange(len(sacct_out))
        #
        return sacct_out
    #
    def get_and_process_sacct_data(self, sacct_str='', max_rows=None, delim=None, with_headers=True, as_dict=False, verbose=False):
        '''
        # fetch and pre-process data from a sacct query (scct_str). Use this by itself or as a worker function.
        # @with_headers: return with headers, as a recarray or structured array, or similar, or just as a block of data.
        # maybe return an object like {'headers':headers 'data':data} ???
        # @with_headers and @as_dict are sort of a confusing mess, but it's what we have for now, and it makes sense when you're
        #   trying to do multiprocessing.:
        #  with_headers==True && as_dict==True: returns dict {'headers':headers, 'data':data}
        #  with_headers==True && as_dict==False: default config, returns a recordarray()
        #  with_headers==False: returns just the data as a list.
        #
        # TODO: large queryies can fail with "results too big" from sacct. Sadly, we can't really anticipate when that will happen.
        #  It's worth seeing if this is memory limited or an actual SACCT restriction.
        #  Possibly, but this is ambitions, add better error handling to split up a query into smaller queries until they succeed.
        #  or just default to shorter activity intervals.
        '''
        #
        if delim is None:
            delim = self.delim
        if delim is None:
            delim = '|'
        if as_dict:
            # as_dict requires headers; returns {'headers': headers, 'data':data}
            with_headers=True
        #
        # NOTE: instead of scctt_str.split(), we can just use the shell=True option.
        sacct_output = subprocess.run(sacct_str.split(), stdout=subprocess.PIPE).stdout.decode().split('\n')
        #print('** ', sacct_str)
        if '--nohheader' in sacct_str:
            headers = [str(k) for k,x in enumerate( numpy.arange(sacct_output[0][:-1].split(delim)) )]
        else:
            headers = sacct_output[0][:-1].replace('\"', '').split(delim)[:-1] + ['JobID_parent']
        #
        if verbose:
            print('*** DEBUG: len(sacct_output): {}'.format(len(sacct_output)))
            print('*** DEGBUG: headers[{}]: {}'.format(len(headers), headers))
        RH = {h:k for k,h in enumerate(headers)}
        #
        data = [self.process_row(rw.replace('\"', ''), headers=headers, RH=RH) for k,rw in enumerate(sacct_output[1:]) if (max_rows is None or k<max_rows) and len(rw)>1 ]
        #
        # NOTE: added this (Very inefficient) step because I though it was failing to sort on the sort fields. It was really failing to sort on an empty set, so I think
        # NOTE: But... it might make sense to sort here and eliminate duplicates. breaking a big query into lots of subquerries
        # will likely produce lots of dupes.
        #   we can skip this.
        # exclude any rows with no submit date or jobid. I hate to nest this sort of logic here, but we seem to get these from time to time. I don't see how they can be valid.
        #  that said... I think this was supposed to fix something that is not actually happening, so we can probably omit this step.
        #data = [rw for rw in data if not (rw[RH['JobID']] is None or rw[RH['JobID']]=='' or rw[RH['Submit']] is None or rw[RH['Submit']])=='' ]
        #
        # pandas.DataFrame(data, columns=active_headers).to_records()
        #
        # this is a bit of a hack, but it should help with MPP.
        #
        if with_headers:
            # TODO: write a to_records() handler, so we don't need to use stupid PANDAS
            if as_dict:
                return {'headers':headers, 'data':data}
            else:
                return pandas.DataFrame(data, columns=headers).to_records()
        else:
            return data
        
        #
    
class SACCT_data_from_h5(SACCT_data_handler):
    # a SACCT_data handler that loads input file(s). Most of the time, we don't need the primary
    #. data; we only want the summary data. We'll allow for both, default to summary...
    #. assume these are in a format that will natively import as a recarray (we'll use recarray->text 
    #. methods to produce them).
    #
    # TODO: rewrite the existing SACCT handler as a base class; rewrite the existing class as a subclass, parallel with this one.
    #  for now, just reproduce a bunch of the __init__() function.
    #
    def __init__(self, h5in_file=None, h5out_file=None, types_dict=None, chunk_size=1000, n_cpu=None,
    n_points_usage=1000, verbose=0, keep_raw_data=False):
        #
        n_cpu = n_cpu or 8
        #
        if h5in_file is None or h5in_file=='':
            raise Exception('Inalid input file.')
        #
        with h5py.File(h5in_file, 'r') as fin:
            #
            # just load everythning? then check for exceptions?
            for ky,ary in fin.items():
                # is this a good idea? this should be lioke self.ky=ary[:], but I always wonder
                #  if hijackeing the internal __dict__ is a bad idea...
                #
                self.__dict__[ky]=ary[:]
            del ky,ary
        #
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})
            #
        #
        ############################
        if types_dict is None:
            types_dict=self.default_types_dict
        # some bits we might need:
        # note: if we don't have jobs_summary, we probably have bigger problems, so let's not go there for now.
        self.headers = self.jobs_summary.dtype.names
        # not sure we need this one, but won't hurt:
        self.RH = {h:k for k,h in enumerate(self.headers)}
        #
        if not 'cpu_usage' in self.__dict__.keys():
            self.cpu_usage = self.active_jobs_cpu(n_cpu=n_cpu, mpp_chunksize=min(int(len(self.jobs_summary)/n_cpu), chunk_size) )
        if not 'weekly_hours' in self.__dict__.keys():
            self.weekly_hours = self.get_cpu_hours(bin_size=7, n_points=n_points_usage, n_cpu=n_cpu)
        if not 'daily_hours' in self.__dict__.keys():
            self.daily_hours = self.get_cpu_hours(bin_size=1, n_points=n_points_usage, n_cpu=n_cpu)
            #
        #
        # especially for dev, allow to pass the recarray object itself...
        if h5out_file is None:
            h5out_file = h5in_file
        self.h5out_file = h5out_file
        #
#
class Tex_Slides(object):
    def __init__(self, template_json='tex_components/EARTH_Beamer_template.json',
                 Short_title='', Full_title='Title',
               author='Mark R. Yoder, Ph.D.',
               institution='Stanford University, School of Earth', institution_short='Stanford EARTH',
              email='mryoder@stanford.edu', foutname='output/HPC_analytics/HPC_analytics.tex',
              project_tex=None ):
        # TODO: modify add_slide() function(s) to accept any correct path to a figure, etc. and resolve the relative path
        #   between the report .tex and the figure (or other). this can be done, more or less like,
        #   my_path = os.path.relpath(os.path.abspath(fig), os.path.abspath(tex) )
        #
        # TODO: test saving/reloading presentation_tex. Also, save inputs, so we can completely reload
        #  an object? Something like Tex_Slides_Obj.json: {project_tex:{}, input_prams:{}}
        #
        # TODO: accept project.json input. this will require handling automated initialization bits, like
        #  (not) automatically computing the header and title sections. it probably makes sense to do all
        #  of that during render() anyway.
        #
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})
        #
        # TODO: add an index. handle user content and ordering with this index. it's separate from
        #. the content dictionary; something simple like [[k, key_k]]. We can nest both in a
        #  JSON format, so we'll load tex_template like, tex_templates=json.load(fin)['templates']
        #  ... but for now, let's not worry about order.
        #
        #foutname_full = os.path.abspath(foutname)
        output_path, output_fname = os.path.split(foutname)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        #
        #
        with open(template_json, 'r') as fin:
            self.tex_templates=json.load(fin)
        #self.header = self.get_header()
        #self.title = self.get_title(Short_title=Short_title, Full_title=Full_title,
        #      author=author, institution=institution, institution_short=institution_short,
        #      email=email )
        #
        # a dict/json object containing the tex components.
        if not project_tex is None:
            if isinstance(project_tex, str):
                # assume it's a filepath:
                with open(project_tex, 'r') as fin:
                    project_tex=json.load(fin)
            #
        else:
            project_tex={}
        #
        # TODO: for a more generalized, data-structured approach, consider a loop with a X[ky]=X.get(ky, f(ky) ) type instruction.
        if not 'header' in project_tex.keys():
            project_tex['header'] = self.get_header()
        if not 'title' in project_tex.keys():
            project_tex['title'] = self.get_title(Short_title=Short_title, Full_title=Full_title,
            author=author, institution=institution, institution_short=institution_short,
            email=email )
        #self.project_tex={'header':self.get_header(), 'title':self.get_title(Short_title=Short_title, Full_title=Full_title,
        #      author=author, institution=institution, institution_short=institution_short,
        #      email=email )}
        #
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})
    #
    @property
    def header(self):
        return self.project_tex['header']
    #
    @property
    def title(self):
        return self.project_tex['title']
    #
    #
    def get_header(self):
        return self.tex_templates['header']

    def get_title(self, Short_title=None, Full_title=None,
           author=None, institution=None, institution_short=None, email=None ):
        '''
        # gets title information from json file
        '''
        #
        return self.tex_templates['title'].format(Short_title=Short_title, Full_title=Full_title, author=author,
                                    institution=institution, institution_short=institution_short,
                                             email=email)
    #
    def add_fig_slide(self, fig_title='Figure', width='.8', fig_path=''):
        #
        # fig_title (aka \frametitle{} ) will complain if it gets an underscore. The "proper" way to write an aunderscore is, \textunderscore, but "\_" should work with
        #  newer versions of latex. I also sometimes just replace "_" -> "-". handing this bluntly here may be a problem, since shouldn't we be able to use a forumula
        #  as a title??? So we shold probably be rigorous and confirm that "_" is (not) wrapped in "$", or handle it on the input side... For now, let's handle the input.
        #  Anyway, we need this to be smart enough to handle corrected input (aka, modify "_" but not "\_")
        #
        rel_path = os.path.relpath(fig_path, self.output_path)
        #
        self.project_tex['slide_{}'.format(len(self.project_tex))] = self.tex_templates['figslide'].format(fig_title=fig_title, width=width, fig_path=fig_path)
        
    #
    def save_presentation_tex(self, foutname="my_presentation.json"):
        with open(foutname, 'w') as fout:
            json.dump(project_tex, fout)
        #
    #
    def render(self, foutname=None):
        if foutname is None:
            foutname=self.foutname
        #
        output_path, fname = os.path.split(foutname)
        #
        # TODO: define relative path something like this...
        rel_path = os.path.relpath(os.path.abspath(output_path), os.path.abspath(self.foutname) )
        fname_root, ext  = os.path.splitext(fname)
        #
        foutname_tex = os.path.join(output_path, f'{fname_root}.tex')
        foutname_pdf = os.path.join(output_path, f'{fname_root}.pdf')
        #
        print(f'*** DEBUG: output_path: {output_path}, fname: {fname}, **: {fname_root}.pdf')
        #
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        #####
        #
        #with open(foutname, 'w') as fout:
        with open(foutname_tex, 'w') as fout:
            fout.write(self.header)
            fout.write('\n')
            fout.write(self.title)
            #
            fout.write('\n\n\\begin{document}\n\n')
            #
            fout.write("\\begin{frame}\n\\titlepage\n\\end{frame}\n\n")
            #
            for ky,val in self.project_tex.items():
                if ky in ('header', 'title'):
                    continue
                #
                print('** adding slide: {}'.format(ky))
                #
                fout.write("\n{}\n".format(val))
            #
            fout.write('\n\n\\end{document}\n\n')
        #
        # not working yet...
        # pdflatex command should be something like:
        #  pdflatex -synctex=1 -interaction=nonstopmode -output-directory=output/HPC_analytics output/HPC_analytics/HPC_analytics.tex
        # ... but the paths need to be handled more carefully. consider constructing full paths to figs.
        #
        #pdf_latex_out = subprocess.run(['pdftex', '-output-directory', output_path, foutname], cwd=None)
        self.pdf_latex_out = subprocess.run(['pdflatex',  f'{fname_root}.tex'], cwd=output_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print('*** pdf_latex status: ', self.pdf_latex_out)
        
    #
#######
class SACCT_groups_analyzer_report(object):

    def __init__(self, Short_title='HPC Analytics', Full_title='HPC Analitics Breakdown for Mazama',
                 out_path='output/HPC_analytics', tex_filename='HPC_analytics.tex', groups=None,
                 add_all_groups=True, n_points_wkly_hrs=500, fig_size=(10,8),
                 fig_width='.8', qs=[.45, .5, .55], SACCT_obj=None, max_rws=None, group_exclusions=group_exclusions ):
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ('self', '__class__')})
        #print('*** DEBUG: __init__: {}'.format(self.out_path))
        #
        if self.groups is None:
            self.groups={'All':list(set(SACCT_obj.jobs_summary['User']))}
        #
        self.make_report()
        
    def make_report(self, out_path=None, tex_filename=None, qs=None, max_rws=None, fig_width=None, n_points_wkly_hrs=None,
    fig_size=(10,8), verbose=1):

        # make some slides, including all the breakdown.
        # inputs: SACCT_data_handler object, groups-dict/JSON,  output_path, cosmetics (like line width, etc.)
        #
        if qs is None:
            qs = self.qs
        if out_path is None:
            out_path = self.out_path
        if tex_filename is None:
            tex_filename = self.tex_filename
        if n_points_wkly_hrs is None:
            n_points_wkly_hrs = self.n_points_wkly_hrs
        groups = self.groups
        if max_rws is None:
            max_rws = self.max_rws
        fig_width = str(fig_width or self.fig_width)
        if fig_size is None:
            fig_size=tuple((12,9))
        #
        SACCT_obj = self.SACCT_obj
        #
        figs_path = os.path.join(out_path, 'figs')
        if not os.path.isdir(figs_path):
            os.makedirs(figs_path)
        #
        HPC_tex_obj = Tex_Slides(Short_title=self.Short_title,
                                         Full_title=self.Full_title,
                                foutname=os.path.join(out_path, tex_filename))
        #
        #
        if isinstance(groups, str):
            with open(groups, 'r') as fin:
                groups = json.load(fin)
        #
        # add an "all" group, with all names from groups. This is an expensive way to "All", since
        #. it wil build and search an index, butwe will benS from simplicity.
        #
        if self.add_all_groups:
            #and not "all" in [s.lower() for s in groups.keys()]:
            groups['All'] = list(set([s for rw in groups.values() for s in rw]))

        print('keys: ', groups.keys() )
        #
        for k, (ky,usrs) in enumerate(groups.items()):
            #
            # DEBUG:
            #if not ky in ('fs-scarp1', 'tgp'): continue
            #
            if ky in self.group_exclusions:
                continue
            if verbose:
                print('*** DEBUG: group: {}'.format(ky))
            #
            # tex corrected group name:
            grp_tex = ky.replace('_', '\_')
            grp_tex = grp_tex.replace('\\_', '\_')
            #
            ix = numpy.where([s in usrs for s in SACCT_obj.jobs_summary['User'] ])
            #
            # any data here?
            if len(SACCT_obj.jobs_summary[ix])==0:
                print('*** WARNING! group: {} no records'.format(ky))
                continue
            #print('** DEBUG: sum(ix)={} / ix.shape={}'.format(numpy.sum(ix), numpy.shape(ix)) )
            if len(ix)==0:
                print('[{}]:: no records.'.format(ky))
                continue
            #
            wkly_hrs = SACCT_obj.get_cpu_hours(bin_size=7, n_points=n_points_wkly_hrs, IX=ix, verbose=0)
            act_jobs = SACCT_obj.active_jobs_cpu(bin_size=None, t_min=None, ix=ix, verbose=0)
            #
            # DEBUG:
            #print('*** weekly: ', wkly_hrs)
            #print('*** act_jobs: ', act_jobs)
            if len(act_jobs)==0:
                print('Group: {}:: no records.'.format(ky))
                continue
            #
            # active jobs/cpus and weekly cpu-hours:
            fg = plt.figure(figsize=fig_size)
            ax1 = plt.subplot(2,1,1)
            ax1.grid()
            ax1a = ax1.twinx()
            ax2 = plt.subplot(2,1,2, sharex=ax1)
            ax2.grid()
            #
            ax1.plot(act_jobs['time'], act_jobs['N_jobs'], ls='-', lw=2., marker='', label='Jobs', alpha=.5 )
            ax1a.plot(act_jobs['time'], act_jobs['N_cpu'], ls='--', lw=2., marker='', color='m',
                      label='CPUs', alpha=.5)
            #ax1a.set_title('Group: {}'.format(ky))
            fg.suptitle('Group: {}'.format(ky), size=16)
            ax1a.set_title('Active Jobs, CPUs')
            #
            ax2.plot(wkly_hrs['time'], wkly_hrs['cpu_hours']/7., ls='-', marker='.', label='bins=7 day', zorder=11)
            #
            ax1.set_ylabel('$N_{jobs}$', size=14)
            ax1a.set_ylabel('$N_{cpu}$', size=14)
            #
            ax1.legend(loc='upper left')
            ax1a.legend(loc='upper right')
            #
            ax2.set_ylabel('Daily CPU hours', size=16)
            #
            fg.canvas.draw()
            #
            # set ax3 labels to dates:
            # now format the datestrings...
            for ax in (ax1,):
                # TODO: FIXME: so... this grossly malfunctions for fs-scarp1. it picks up the second positional argument as text, instead of the first. aka:
#                ticklabels:  [Text(737445.5269299999, 0, '0.000030'), Text(737445.526935, 0, '0.000035'), Text(737445.5269399999, 0, '0.000040'), Text(737445.5269449999, 0, '0.000045'), Text(737445.52695, 0, '0.000050'), Text(737445.5269549999, 0, '0.000055'), Text(737445.5269599999, 0, '0.000060')]
#                ticklabels:  ['0.000030', '0.000035', '0.000040', '0.000045', '0.000050', '0.000055', '0.000060']
#                from this:
#                print('\nticklabels: ', [s for s in ax.get_xticklabels()])
#                print('ticklabels: ', [s.get_text() for s in ax.get_xticklabels()])
                #
                #if ky in ('fs-scarp1'):
                #    continue
                #
                # just trap this so that it works. could throw a warning...
                try:
                    lbls = [simple_date_string(mpd.num2date( float(s.get_text())) ) for s in ax.get_xticklabels()]
                except:
                    print('** WARNING: failed writing date text labels:: {}'.format('lbls = "[simple_date_string(mpd.num2date(max(1, float(s.get_text())) ) ) for s in ax.get_xticklabels()]" '))
                    print('"*** SysInfo: {}'.format(sys.exc_info()[0]))
                    print('*** trying to write x-ticks with failsafe mmpd.num2date(max(1, float(s.get_text())) )')
                    lbls = [simple_date_string(mpd.num2date(max(1, float(s.get_text())) ) ) for s in ax.get_xticklabels()]
                #
                ax.set_xticklabels(lbls)
            #
            #
            # Save figure and add slide:
            cpu_usage_fig_name = os.path.join('figs', '{}_cpu_usage.png'.format(ky))
            cpu_usage_fig_path=os.path.join(out_path, cpu_usage_fig_name)
            plt.savefig(cpu_usage_fig_path)
            HPC_tex_obj.add_fig_slide(fig_title='{}: CPU/Jobs Requests'.format(grp_tex),
                                                width=fig_width, fig_path=cpu_usage_fig_name)
            print('*** [{}] Slide_1 added??:: {}'.format(ky, len(HPC_tex_obj.project_tex)))

            #
            # now, add the active jobs report:
            #jobs_per_name=os.path.join('figs', '{}_jobs_per.png'.format(ky))
            #jobs_per_path=os.path.join(output_path, jobs_per_name)
            #
            zz = SACCT_obj.active_cpu_jobs_per_day_hour_report(qs=qs,
                                                figsize=fig_size, cpu_usage=act_jobs,foutname=None, periodic_projection='polar')
            plt.suptitle('Instantaneous Usage: {}'.format(ky), size=16)
            #
            #
            jobs_per_name=os.path.join('figs', '{}_jobs_per.png'.format(ky))
            jobs_per_path=os.path.join(out_path, jobs_per_name)
            #
            plt.savefig(jobs_per_path)
            HPC_tex_obj.add_fig_slide(fig_title='{}: Periodic Usage'.format(grp_tex),
                                                width=fig_width, fig_path=jobs_per_name)
            #
            print('*** Slide_2 added??:: ', len(HPC_tex_obj.project_tex))
            #print('** DEBUG: dpu_usage_fig_path:: {}'.format(cpu_usage_fig_path))
            #print('** DEBUG: jobs_per_path:: {}'.format(jobs_per_path))
            #
            #
            if (not max_rws is None) and k >= max_rws:
                break
            #HPC_tex_obj.render()
            #
        self.HPC_tex_obj=HPC_tex_obj
        HPC_tex_obj.render()
        #
    #
#
class SACCT_groups_analyzer_report_handler(object):
    '''
    # As best as I can tell, this is intended to be a generalized extension of the earlier SACCT_groups_analyzer_report() class, but designed
    # to host multiple report types. Reports can nominally be run externally (eg, create this object, then execute plot functions --
    # both internal to this Class or external, and add slides to a report, or reports can be coded into the class itself.
    '''
    #
    def __init__(self, Short_title='HPC Analytics', Full_title='HPC Analitics Breakdown for Mazama',
                 out_path='output/HPC_analytics', tex_filename='HPC_analytics.tex', n_points_wkly_hrs=500,
                 fig_width_tex='.8', qs=[.45, .5, .55], fig_size=(10,8), SACCT_obj=None, TEX_obj=None, max_rws=None ):
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ('self', '__class__')})
        #print('*** DEBUG: __init__: {}'.format(self.out_path))
        #
        #if self.groups is None:
        #    self.groups={'All':list(set(SACCT_obj.jobs_summary['User']))}
        #
        #self.HPC_tex_obj = Tex_Slides(Short_title=self.Short_title,
        #         Full_title=self.Full_title,
        #         foutname=os.path.join(out_path, tex_filename))
        #
        self.HPC_tex_obj = (TEX_obj or Tex_Slides(Short_title=self.Short_title,
                 Full_title=self.Full_title,
                 foutname=os.path.join(out_path, tex_filename)) )
    #
    def standard_reports_slides(self, ix=None, out_path=None, group_name='group', qs=None, fig_width_tex=None ):
        '''
        # Standard set of slides for a sub-group, defined by the index ix
        '''
        #
        # Standard frontmatter bits:
        # NOTE: except for diagnostic purposes, this should probably usually be None and default to the parent object.
        #  otherwise, it becomes difficult to stitch together the .tex object. Namely, it becomes necessary to use either
        #  absolute paths for figure refrences, which are not portable, or really complex navigation of relative paths --
        #  which maybe Python can do for us? we'd maybe have a figure path like ../../p1/p2/figname.png.
        #
        if out_path is None:
            out_path =self.out_path
            #out_path = os.path.join(self.out_path, 'figs')
        out_path_figs = os.path.join(out_path, 'figs')
        #
        if not os.path.isdir(out_path_figs):
            os.makedirs(out_path_figs)
        #
        # this is mostly to remind us that this part is still messy. this is the figs path to write into the tex file.
        # for this to be rigorous, we must either 1) not allow an out_path input, 2) use an absolut path, or 3) do a difficult navigation
        #  of the relative path, all of which are messy or limit debugging options, etc. so we'll maybe let this break sometimes...
        out_path_figs_tex = 'figs'
        #
        qs = qs or self.qs
        fig_width_tex = (fig_width_tex or self.fig_width_tex)
        fig_size=self.fig_size
        #
        ###################################
        #
        # tex corrected group name:
        group_name_tex = group_name.replace('_', '\_')
        group_name_tex = group_name_tex.replace('\\_', '\_')
        #
        # make figures and add slides:
        activity_figname       = '{}_activity_ts.png'.format(group_name)
        periodic_figname_rect  = '{}_periodic_usage_rect.png'.format(group_name)
        periodic_figname_polar = '{}_periodic_usage_polar.png'.format(group_name)
        #
        activity_figpath = os.path.join(out_path_figs, activity_figname )
        periodic_figpath_rect = os.path.join(out_path_figs, periodic_figname_rect )
        periodic_figpath_polar = os.path.join(out_path_figs, periodic_figname_polar )
        #
        act_fig = self.activity_figure(ix=ix, fout_path_name=activity_figpath, group_name=group_name)
        per_fig = self.periodic_usage_figure(ix=ix, fout_path_name=periodic_figpath_rect, qs=qs, fig_size=fig_size,
                group_name=group_name)
        per_fig_polar = self.periodic_usage_figure(ix=ix, fout_path_name=periodic_figpath_polar, qs=qs, fig_size=fig_size,
                group_name=group_name, projection='polar' )
        #
        # add figures to report:
        self.HPC_tex_obj.add_fig_slide(fig_title='{}: CPU/Jobs Requests'.format(group_name_tex),
            width=fig_width_tex, fig_path=os.path.join(out_path_figs_tex, activity_figname) )
        #
        self.HPC_tex_obj.add_fig_slide(fig_title='{}: Periodic usage'.format(group_name_tex),
            width=fig_width_tex, fig_path=os.path.join(out_path_figs_tex, periodic_figname_rect))
        #
        self.HPC_tex_obj.add_fig_slide(fig_title='{}: Periodic Clock-Plots'.format(group_name_tex),
            width=fig_width_tex, fig_path=os.path.join(out_path_figs_tex, periodic_figname_polar))
        #
    #
    def activity_figure(self, ix=None, fout_path_name=None, qs=None, fig_size=None, n_points_wkly_hrs=None, group_name='group', verbose=0):
        # images then slides for just one group, which we define from an index.
        #
        # TODO:
        # how is ix=None handled in called functions? ideally, we want to avoid passing a large index if possible.
        if qs is None:
            qs = self.qs
        fig_size = (fig_size or self.fig_size)
        #
        if fout_path_name is None:
            out_path = self.out_path
            #cpu_usage_fig_name = os.path.join('figs', '{}_cpu_usage.png'.format(ky))
            fname = 'cpu_usage.png'
            fout_path_name = os.path.join(out_path, fname)
        #
        if n_points_wkly_hrs is None:
            n_points_wkly_hrs = self.n_points_wkly_hrs
        #
        # compute figures:
        wkly_hrs = self.SACCT_obj.get_cpu_hours(bin_size=7, n_points=n_points_wkly_hrs, IX=ix, verbose=0)
        act_jobs = self.SACCT_obj.active_jobs_cpu(bin_size=None, t_min=None, ix=ix, verbose=0)
        #
        if len(act_jobs)==0:
            print('Group: {}:: no records.'.format(group_name))
            #print('Group: :: no records.' )
            #continue
            # TODO: return empty set?
            return None
        #
        # active jobs/cpus and weekly cpu-hours:
        fg = plt.figure(figsize=fig_size)
        ax1 = plt.subplot(2,1,1)
        ax1.grid()
        ax1a = ax1.twinx()
        ax2 = plt.subplot(2,1,2, sharex=ax1)
        ax2.grid()
        #
        ax1.plot(act_jobs['time'], act_jobs['N_jobs'], ls='-', lw=2., marker='', label='Jobs', alpha=.5 )
        ax1a.plot(act_jobs['time'], act_jobs['N_cpu'], ls='--', lw=2., marker='', color='m',
                  label='CPUs', alpha=.5)
        #
        fg.suptitle('Group: {}'.format(group_name), size=16)
        ax1a.set_title('Active Jobs, CPUs')
        #
        ax2.plot(wkly_hrs['time'], wkly_hrs['cpu_hours']/7., ls='-', marker='.', label='bins=7 day', zorder=11)
        #
        ax1.set_ylabel('$N_{jobs}$', size=14)
        ax1a.set_ylabel('$N_{cpu}$', size=14)
        #
        ax1.legend(loc='upper left')
        ax1a.legend(loc='upper right')
        #
        ax2.set_ylabel('Daily CPU hours', size=16)
        #
        fg.canvas.draw()
        #
        # set ax3 labels to dates:
        # now format the datestrings...
        for ax in (ax1,):
            #print('*** xticklabels(): {}'.format([s.get_text() for s in ax.get_xticklabels() ] ) )
            #print('*** xcticks(): {}'.format( ax.get_xticks() ) )
            #print('*** lbls_prim: {}'.format([simple_date_string(mpd.num2date(max(1, float(s.get_text())) ) ) for s in ax.get_xticklabels()]) )
            #print('*** act_jobs[time]: {}'.format(act_jobs['time']))
            #
            #break
            # TODO: FIXME: so... this grossly malfunctions for fs-scarp1. it picks up the second positional argument as text, instead of the first. aka:
#                ticklabels:  [Text(737445.5269299999, 0, '0.000030'), Text(737445.526935, 0, '0.000035'), Text(737445.5269399999, 0, '0.000040'), Text(737445.5269449999, 0, '0.000045'), Text(737445.52695, 0, '0.000050'), Text(737445.5269549999, 0, '0.000055'), Text(737445.5269599999, 0, '0.000060')]
#                ticklabels:  ['0.000030', '0.000035', '0.000040', '0.000045', '0.000050', '0.000055', '0.000060']
#                from this:
#                print('\nticklabels: ', [s for s in ax.get_xticklabels()])
#                print('ticklabels: ', [s.get_text() for s in ax.get_xticklabels()])
            #
            #if ky in ('fs-scarp1'):
            #    continue
            #
            # just trap this so that it works. could throw a warning...
            try:
                lbls = [simple_date_string(mpd.num2date( float(s) ) ) for s in ax.get_xticks()]
            except:
                print('** WARNING: failed writing date text labels:: {}'.format('lbls = "[simple_date_string(mpd.num2date(max(1, float(s.get_text())) ) ) for s in ax.get_xticklabels()]" '))
                print('"*** SysInfo: {}'.format(sys.exc_info()[0]))
                print('*** trying to write x-ticks with failsafe mmpd.num2date(max(1, float(s.get_text())) )')
                lbls = [simple_date_string(mpd.num2date( float(s) ) ) for s in ax.get_xticks()]
            #
            ax.set_xticklabels(lbls)
        #
        # Save figure:
        plt.savefig(fout_path_name)
    #######
    #
    def periodic_usage_figure(self, ix=None, fout_path_name=None, qs=None, fig_size=None,
            group_name='group', projection=None):
        '''
        # periodic usage (aka, jobs per hour-of-day, etc.)
        #
        '''
        #
        if fig_size == None:
            fig_size=self.fig_size
        #
        if qs is None:
            qs = self.qs
        #
        if fout_path_name is None:
            out_path = self.out_path
            #
            fname = 'periodic_usage.png'
            fout_path_name = os.path.join(out_path, fname)
        #
        # TODO: is there a beter way to handle the image. here, we use plt. to get the (implicit) current figure/axis object,
        #  but this is kind of a dumb way to do this -- it would be better to have a defined figure or axis instance. return one? add a parameter
        #  for suptitle(), and other stuff too? or just don't add suptitle() and pass a foutname().
        #
        #act_jobs = SACCT_obj.active_jobs_cpu(bin_size=None, t_min=None, ix=ix, verbose=0)
        zz = self.SACCT_obj.active_cpu_jobs_per_day_hour_report(qs=qs,
                                            figsize=fig_size,
                                            cpu_usage=self.SACCT_obj.active_jobs_cpu(bin_size=None, t_min=None, ix=ix, verbose=0),
                                            foutname=None, periodic_projection=projection)
        plt.suptitle('Periodic Usage: {}'.format(group_name), size=16)
        #
        plt.savefig(fout_path_name)
#
def day_of_week_distribution(time=None, Y=None, qs=[.25, .5, .75, .9]):
    # TODO: add a DoW figure, with ax input.
    #
    dtype_qs = [('DoW', '>i8')] + [(f'q{k+1}', '>f8') for k,x in enumerate(qs)]
    DoWs = numpy.zeros((7,), dtype=dtype_qs)
    DoWs['DoW'] = numpy.arange(7)
    #
    # I think you can do this for a recarray, but not for a structured array? I know this just worked somewhere...
    #DoWs.qs=qs
    #
    if isinstance(time[0], dtm.datetime) or isinstance(time[0], numpy.datetime64):
        time_working = mpd.date2num(time)
    else:
        # assume float/number. if it's not, it will probably break...
        time_working =  time[:]
    #
    # this is not the most efficient or fastest way to do this. The fastest is to just do t_flopat%7-4
    #days = [x.weekday() for x in time_working]
    days = numpy.mod(time_working+3, 7).astype(int)
    #
    for k in DoWs['DoW']:
        q_vals = numpy.nanquantile(Y[days==k], qs)
        for j,q in enumerate(q_vals):
            DoWs[f'q{j+1}'][k]=q
            #
        #
    #
    return DoWs
    
def hour_of_day_distribution(time=None, Y=None, qs=[.25, .5, .75, .9]):
    dtype_qs = [('hour', '>i8')] + [(f'q{k+1}', '>f8') for k,x in enumerate(qs)]
    output = numpy.zeros((24,), dtype=dtype_qs)
    output['hour'] = numpy.arange(len(output))
    #
    # I think you can do this for a recarray, but not for a structured array? I know this just worked somewhere...
    #DoWs.qs=qs
    #
    if isinstance(time[0], dtm.datetime) or isinstance(time[0], numpy.datetime64):
        time_working = mpd.date2num(time)
    else:
        # assume float/number. if it's not, it will probably break...
        time_working =  time[:]
    #
    # this is not the most efficient or fastest way to do this. The fastest is to just do t_flopat%7-4
    #days = [x.weekday() for x in time_working]
    hours = (24.*numpy.mod(time_working, 1.)).astype(int)
    #
    for k in output['hour']:
        q_vals = numpy.nanquantile(Y[hours==k], qs)
        for j,q in enumerate(q_vals):
            output[f'q{j+1}'][k]=q
            #
        #
    #
    return output
#
def get_group_users(user_id):
    # # and get a list of users to construct an index:
    # $ finger dunham
    # Login: edunham         Name: Eric Dunham
    # Directory: /home/edunham             Shell: /bin/bash
    # Never logged in.
    # No mail.
    # No Plan.
    # [rcwhite@cees-mgmt0 ~]$ id edunham
    # uid=60367(edunham) gid=100(users) groups=100(users),203(tgp),70137(fs-erd)
    # [rcwhite@cees-mgmt0 ~]$ getent group | grep tgp
    # tgp:*:203:ooreilly,kashefi,malmq,axelwang,lwat054,glotto,chao2,bponemon,danmohad,sinux1,
    # gnava,eliasrh,dennis,zhuwq,yyang85,sbydlon,houyun,cstierns,mrivet,jlmaurer,myoder96,sozawa,schu3,
    # lbruhat,kallison,labraha2,kcoppess,edunham
    #
    # to get all groups:
    # $ groups
    # users tgp sep clab beroza astro oneill modules itstaff suprib cees suckale schroeder thomas ere mayotte-collab
    #
    # first, get basic info:
    # capture_output=True is a later Python version syntax (maybe Python 3.7+ ? Mazama runs 3.6, at least for standard Anaconda).
    user_info_call_seq = ['id', user_id]
    #user_info = subprocess.run(user_info_call_seq, capture_output=True)
    #
    #print('** input: ', user_info_call_seq)
    S = subprocess.run(user_info_call_seq, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    print('*** S.stdout: ', S.stdout.decode() )
    #
    S = S.stdout.decode()[:-1].split(chr(32))
    S = dict([rw.split('=') for rw in S])
    #
    return S
#
#
def get_resercher_groups_dict():
    '''
    # get research groups and members, for Mazama. This will (probably) not work correctly on Sherloc, and in any case, there are better
    #  ways to do this on Sherlock.
    '''
    # TODO: constraints for this? I think this was developed for Mazama. For Sherlock, we might need some secodary groups, or
    #  some other constraint.
    sp_command = 'getent group'
    #
    sp = subprocess.run(shlex.split(sp_command), shell=False, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    sp_out = {}
    for rw in sp.stdout.decode().split('\n'):
        rws = rw.split(':')
        if len(rws)<4:
            continue
        #
        if rws[1]!="*":
            continue
        #
        usrs = rws[3].split(',')
        if len(usrs)==0:
            continue
        #
        for u in user_exclusions:
            while u in usrs:
                usrs.remove(u)
            #
        #
        sp_out[rws[0]] = usrs
    #
    return sp_out
#
def get_PI_groups_from_groups(groups='sh_s-ees', user_list=[]):
    '''
    # get a list of PI (primary) groups for all members of a secondary group. For example, get PI groups for all members of a shared partition or school.
    # both groups are linux groups. Alternatively pass a user_list; get all unique primary groups for those users.
    # @groups: a linux group id (or list of)
    # @user_list: Optional. List of user IDs.
    # the two lists will be added together.
    '''
    #
    #
    if not groups is None:
        for sg in numpy.atleast_1d(groups):
            #print('*** *', ' '.join(['getent', 'group', sg]))
            user_list += subprocess.run(['getent', 'group', sg], shell=False, capture_output=True).stdout.decode().replace('\n','').split(',')
    #
    # Now, collect into a {PI_group:]users], ...} dict.
    pi_groups={}
    for usr in user_list:
    #for pi, usr in zip([subprocess.run(['id', '--group', '--name', s], shell=False, capture_output=True).stdout() for s in user_list], user_list):
        #print('*** usr: ', usr)
        if "*" in usr:
            continue
        #
        pi = subprocess.run(['id', '--group', '--name', usr], shell=False, capture_output=True).stdout.decode().replace('\n', '')
        #
        
        if not pi in pi_groups.keys():
            pi_groups[pi]=[]
        #
        pi_groups[pi] += [usr]
    #
    return pi_groups

def get_group_members(group_id):
    '''
    # use a subprocess call to getent to get all members of linux group group_id
    '''
    #
    #
    #sp_command="getent group | grep {}".format(group_id)
    #sp_command = f'getent group {group_id}'
    sp_command = ['getent', 'group', group_id]
    #
    # NOTE: in order to run a complex linux command, set shell=True. Some will say, however, that this is not recommended for security -- among other, reasons.
    # https://stackoverflow.com/questions/13332268/how-to-use-subprocess-command-with-pipes
    # The stackoverflow recommendation is to basically use two subprocesses (in this case, the first for getent, and then pipe the stdout to the stdin of a second
    #  process. For something simple, I don't see the advantage of that over just doing the grep or other operation in Python, which is a better environment for that
    #  sort of work anway. The advantage of the piped command is it is only a single call to the OS.
    #
    # Python 3.6 might still require this syntax:
    #sp = subprocess.run(shlex.split(sp_command), shell=False, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    # this should work for python 3.8 or so
    sp = subprocess.run(sp_command, shell=False, capture_output=True)
    #
    return sp.stdout
#
def active_groups_to_json(fout_name='mazama_groups.json'):
    # TODO: (maybe?) return groups{}, so the user can verify?
    #groups = get_resercher_groups_dict()
    #
    with open(fout_name, 'w') as fout:
        json.dump(get_resercher_groups_dict(), fout)
    #
    return None
#
def time_bin_aggregates(XY, bin_mod=24, qs=numpy.array([.25, .5, .75])):
    # NOTE: this is not quite general purpose. it takes the input%1, then converts the fractional remainder
    #. (modulus) to an integer(ish) by multiplying. Aka, t%1 gives the remaining fraction of a day (by standard
    #. python date conventions); (t%1)*24 gives that in hours. But to convert this to DoW, we first have 
    #. to convert the numerical date to weeks, so we'd want (t-t_0)%7; to use this code we
    #. pass t=t_days/7., bin_mod=7, and really we'd want to do a phase shif to get DoW correctly.
    #  note that num_to_date(0) gives a Thursday, so we also need to shift DoW -=3 to get MTWTFSS correctly.
    #
    XY=numpy.array(XY)
    if XY.shape[0]==2:
        X = XY[0,:]
        Y = XY[1:]
    else:
        X = XY[:,0]
        Y = XY[:,1]
    #
    #X_mod = ((X*bin_mod)%bin_mod).astype(int)
    X_mod = ((X%1.)*bin_mod).astype(int)
    #X_out = numpy.zeros( (len(numpy.unique(X_mod)), 3+len(qs) ), dtype=[('x', '>f8'), ('mean', '>f8'),
    #                                                    ('stdev', '>f8')] + 
    #                                     [('q_{}'.format(q), '>f8') for q in qs])
    #
    #print('** X_out: ', X_out, X_out.dtype)
    #print('** shape: ', X_out.shape)
    stats_output=[]
    for k,x in enumerate(numpy.unique(X_mod) ):
        ix = X_mod==x
        this_Y = Y[ix]
        #stats_output += [numpy.append([x, numpy.mean(this_Y), numpy.std(this_Y)],
        #                              numpy.quantile(this_Y, qs))]
        stats_output += [numpy.append([x, numpy.mean(this_Y), numpy.std(this_Y)],
        numpy.percentile(this_Y, 100.*numpy.array(qs)))]
        #
        # ... I confess that assignment to these structured arrays is baffling me...
        #X_out[k,:] = numpy.append([x, numpy.mean(this_Y), numpy.std(this_Y)],
        #                                  numpy.quantile(this_Y, qs))[:]
    # TODO: convert this to a structured array. it looks (mostly) just like a record array, but it's not...
    #.  and it's faster...
#     return numpy.core.records.fromarrays(numpy.array(stats_output).T, dtype=[('x', '>f8'), ('mean', '>f8'),
#                                                         ('stdev', '>f8')] + 
#                                          [('q_{}'.format(q), '>f8') for q in qs])
    #
    # the syntax for converting a list or array to a structured array still appears to be clunky, but (unless something
    #. has changed, or depending on the numpy version (??), the performance benefit of structured arrays is significant.
    #
    # this works, but is maybe a bit overhead heavy?
    return numpy.array([tuple(rw) for rw in stats_output], dtype=[('x', '>f8'), ('mean', '>f8'),
                                                        ('stdev', '>f8')] + 
                                         [('q_{}'.format(q), '>f8') for q in qs])
#
# helper functions:
@numba.jit
def process_sacct_row(rw, delim='\t', headers=None, types_dict={}, RH={}):
    # use this with MPP processing:
    # ... but TODO: it looks like in the Class() scope, this is 1) inefficient and 2) breaks with large data inputs because I think it pickles the entire
    #  class object... so we need to move the MPP object out of class.
    #  for now, let's assume that the map() (or whatever) will consolidate and (pseudo-)vectorize. Maybe this should be a Process() object....
    #  or at lest a small class, initialized with all but rw (the iterable), so we can call it easily wiht map_async()
    #
    # use this for MPP processing:
    rws = rw.split(delim)
    #return [None if vl=='' else self.types_dict.get(col,str)(vl)
    #            for k,(col,vl) in enumerate(zip(self.headers, rw.split(self.delim)[:-1]))]
    return [None if vl=='' else types_dict.get(col,str)(vl)
                for k,(col,vl) in enumerate(zip(headers, rws[:-1]))] + [rws[RH['JobID']].split('.')[0]]
#
@numba.jit
def get_modulus_stats(X,Y, a_mult=1., a_mod=1., qs=numpy.array([.5, .75, .95])):
    '''
    # NOTE: in retrospect, this simple script has its moments of usefulness, but a lot of these cases still need
    #  to be custom coded to get the right type of aggregation. In many cases -- namely where we are just counting
    #  in some modulo bin, we need to define a second binning over which to do averaging (ie, k_week = k//k0, k_day=k%k0),
    #  and we'd want to aggregate for each k_week, all of k_day. This can be expensive, so we might want to come up with
    #  something simpler.
    #
    # stats of Y on X' = (x*a_mult)%a_mod.
    # for example, to get hour-of-day from a X=mpd.date2num() sequence (where the integer unit is days), HoD=(X*24)%24.
    #  to get Day of Week, DoW = (X*1)%7
    #
    # of course, this can be done with just a fraction (a_mult=1, a_mod={not-integer), but this can introduce numerical error
    #  and is less intuitive to some.
    '''
    #
    X_prime = (numpy.array(X)*a_mult)%a_mod
    #
    # note: this will come back sorted.
    X0 = numpy.unique(X_prime)
    #
    # TODO: consider sorting and then k_j=index(X_sorted,j) -- or something like that.
    #  a night-n-day performance improvement is not obvious; there are a couple of loops through
    #  the vector to do this, and part of it has to be done outside of vectorization if we are to avoid
    #  re-scanning the early part of the vector. so we'll save that for a later upgrade.
    #
    # ... and there probably is a way to do this by broadcasting over unique indices and then aggregating
    #  over an axis, but it will cost some memory to do so, and the savings -- again , is not clear.
    # ... and memory has been a problem in the past, so for now we'll walk a line with memory efficiency as a real consideration.
    #
    quantiles = numpy.array([numpy.percentile(Y[X_prime==j], 100.*numpy.array(qs)) for j in X0])
    means = numpy.array([numpy.mean(Y[X_prime==j], 100.*numpy.array(qs)) for j in X0])
    #
    return numpy.core.records.fromarrays(numpy.append([X0, means], quantiles.T, axis=0),
             dtype=[('time', '>f8'), ('mean', '>f8')] +
            [('q{}'.format(k), '>f8') for k,q in enumerate(qs)] )
#
def array_to_hdf5_dataset(input_array, dataset_name, output_fname='outfile.h5', h5_mode='a', verbose=0):
    '''
    # copy input_array into a datset() object inside an hdf5 File object (aka, hdf5_handler.create_dataset(dataset_name, numpy.shape(input_array) )
    # @input_array: a numpy structured- or rec-array (like) object
    # @dataset_name: (string) name of the data set.
    # @output_fname: hdf5 output file
    # @h5_mode: file open mode, aka with hdf5.File(dataset_name, h5_mode):
    #
    '''
    #
    # need to handle if dtype, but for now just assume (or change this function to assume):
    new_dtype = []
    for nm, tp in input_array.dtype.descr:
        new_type = tp
        #  or tp.startswith('|S') ??
        if tp == '|O':
            #print('*** converting: [{}], {}, {} '.format(nm, tp, input_array[nm][0:20]) )
            new_type = 'S{}'.format(numpy.max([len(fix_to_ascii(str(s))) for s in input_array[nm] ]) )
        #
        # we might actually be able to get rid of this condition. These are HDF5 definitions,
        #  so they might work fine here...
        if isinstance(tp, tuple):
            # TODO: we get advanced types that include encoding information, like:
            #  ('User', ('|S8', {'h5py_encoding': 'ascii'}))
            #   We shoule (eventually) handle th encoding more rigorously.
            #
            new_type=(tp[0])[1:]
        #
        new_dtype += [(nm, new_type)]
    #
    if verbose:
        print('*** ndt', new_dtype)
    #
#     print('** ', sacct_obj.jobs_summary.dtype.descr)
#
    with h5py.File(output_fname, h5_mode) as fout:
        #fout.create_dataset('jobs_summary', data=sacct_obj.jobs_summary,
        # TODO: try the syntax like:
        # dataset[...] = some_2d_array
        #                    dtype=new_dtype)
        #
        # this works, but seems clumsy...
        ds = fout.create_dataset(dataset_name, input_array.shape,
                            dtype=new_dtype)
        #
        if verbose:
            print('*** DEBUG: dataset created; update data.')
        #for k,cl in enumerate(input_array.dtype.names):
        for k, (cl, tp) in enumerate(ds.dtype.descr):
            print('*** DEBUG: updating column [{}], type={}'.format(cl, tp))
            #if tp.startswith('S'):
            if isinstance(tp, tuple):
                # probably a string-like type, or maybe a blob or something.
                # can I .encode() the whole array at once?
                # they appear to look like this:
                # type=('|S8', {'h5py_encoding': 'ascii'})
                #
                # should be able to do something like one of these:
                #ds[cl]=input_array[cl][:].astype(tp[1]['h5py_encoding'])
                #ds[cl]=input_array[cl][:].astype(numpy.str)
                #
                #ds[cl]=[s.encode(tp[1].get('h5py_encoding', 'ascii') ) for s in input_array[cl]]
                ds[cl]=[fix_to_ascii(s) for s in input_array[cl]]
                
            else:
                ds[cl]=input_array[cl][:]
        #
    #
    return None
#
def fix_to_ascii(s):
    if s is None:
        return ''
    if isinstance(s,bytes):
        s = s.decode()
    #
    # TODO: better string handling please...
    chrs = {8211:'--', 2013:'_', 8722:'-'}
    #
    for k,c in chrs.items():
        #
        s = s.replace(chr(k),c)
    #
    # rigorously,we might need something like this:
    out_str = ''
    for c in s:
        #
        #cc = chrs.get(ord(c), c)
        #for c1 in cc:
        #    out_str += chr(max(127, ord( c1) ) )
        out_str += chr(min(127, ord(c)))
    #
    # but let's see if it will be necessary, or if there's a smarter way to do it.
    #
    return out_str
    
def calc_jobs_summary(data=None, verbose=0, n_cpu=None, step_size=1000):
    '''
    # compute jobs summary from (raw)data
    # @n_cpu: number of comutational elements (CPUs or processors)
    # @step_size: numer of rows per thread (-like chunks). We'll use a Pool() (probably -- or something) to do the computation. use this
    #   variable to break up the job into n>n_cpu jobs, so we don't get a "too big to pickle" error. we'll also need to put in a little
    #  intelligence to e sure the steps are not too small, all distinct job_ids are represented, etc.
    #
    #  TODO: and...
    #  NOTE: One problem we see is redundancy due to small sub-queries. Specifically, SACCT returns data for 'all jobs in any state'
    #   between the start/end dates. This means that if we break up the query, which we must 1) for performance, 2) SLURM restrictions
    #   on number of rows returned, 3) Sherlock/Kilian restrictions on query bredth. The general mechanism to summarize jobs *should*
    #   also de-dupe these rows (sort by job_id, then ???; then take max/min values for start, end, etc. times). BUT, this seesm to be
    #   making mistakes.
    '''
    #
    #
    if data is None:
        raise Exception('No data provided.')
    #
    # TODO: this:
#    if hasattr(data, 'dtype'):
#        dtype_dict = dict(list(data.dtype))
#        # or:
#        #dtype_dict = {ky:val for ky,vl in data.dtype}
    #
    n_cpu = (n_cpu or 1)
    #print('*** DEBUG n_cpu: {}'.format(n_cpu))
    if verbose:
        print('*** computing jobs_summary func on {} cpu'.format(n_cpu))
    #
    if n_cpu > 1:
        # this should sort in place, which could cause problems downstream, so let's use an index:
        print('*** DEBUG: data[JobID]: {}, data[Submit]: {}'.format( (None in data['JobID']), (None in data['Submit']) ))
        #
        # Getting some "can't sort None type" errors:
        print('*** DEBUG: type(data), len(data): ', type(data), len(data) )
        #print('*** DEBUG: nans in JobID?: ', numpy.isnan(data['JobID']).any() )
        print('*** DEBUG: nans in Submit?: ', numpy.isnan(data['Submit']).any() )
        print('*** DEGUG: Nones in JobID: ', numpy.sum([s is None for s in data['JobID']]))
        #
        # NOTE: sorts by named order=[] fields, then in order as they appear in dtype, to break any ties. When there are LOTS of records,
        #  and more pointedly, when there are dupes resulting from breaking up into sub-querries, all the real fields will tie,
        #  and eventually it will attempt to sort on a Nonetype which will break. The workaround is to add an index column, which is
        #  necessarily unique to a data set
        #sort_kind='stable'
        sort_kind='quicksort'
        #ix_s = numpy.argsort(data, axis=0, order=['JobID', 'Submit', 'index'], kind=sort_kind)
        #
        #
        #working_data = data[ix_s]
        working_data = data[:]
        working_data['index'] = numpy.arange(len(working_data))
        #
        # FIXME: This is likely the problem we are seeing when we break up a query. A couple things: we are a little inconsistent
        #  with use of JobID vs JobID_parent, but this is usually a non-problem since JobID_parent is a derived field
        #  that just truncates the parent jobID from Arrays like, JobID_parent=JobID.split('.')[0], so
        #  12345.67 -> 12345.
        #  The bigger problem is that we get duplicates when we split up a large query into smaller sub-queries, so eventually
        #  the algorithm tries to sort on a NONEtype field. We can fix this by adding an index. We do appear to be adding
        #  this 'index' column to the sacct data, so let's use that!
        if verbose:
            print('*** DEBUG: working_data.dtype: {}'.format(working_data.dtype))
            print('*** DEBUG: len(working_data[index]), len(unique(working_data[index])): {}, {}'.format(len(working_data['index']), len(numpy.unique(working_data['index']))))        #working_data.sort(axis=0, order=['JobID', 'Submit'], kind=sort_kind)
        #
        # keep getting nonetype! seems like sorting on 'index' should pretty much sort it out, but no...
        #   this is possibly best called a "bug." It would seem that sort() sorts along the specified axes and then all the
        #   rest of them anyway (even after the sort is resolved). it is worth noting then that this might also result in a
        #   performance hit, so even though our solution -- to compute an index, looks stupid, it might actually be best practice.
        # seems sort of stupid, but this might work.
        #ix_temp = numpy.argsort(working_data[['JobID_parent', 'index']], order=['JobID_parent', 'index'])
        working_data = working_data[numpy.argsort(working_data[['JobID_parent', 'index']], order=['JobID_parent', 'index'])]
        #working_data.sort(order=['JobID_parent', 'index'], kind=sort_kind)
        #
        #ks = numpy.append(numpy.arange(0, len(data), step_size), [len(data)+1] )
        ks = numpy.array([*numpy.arange(0, len(data), step_size), len(data)+1])
        #print('*** *** ks_initial: ', ks)
        #
        # TODO: some sorting...
        # Adjust partitions to keep jobs together.
        for j,k in enumerate(ks[1:-1]):
            #print('*** k_in: ', k)
            while working_data['JobID_parent'][k] == working_data['JobID_parent'][k-1]:
                #print('** ** ', working_data['JobID_parent'][k], working_data['JobID_parent'][k-1])
                k+=1
            ks[j+1] = k
            #print('*** k_out: {}/{}'.format(k, ks[j]))
        ks = numpy.unique(ks)
        if verbose:
            print('** calc_summary:: ks: {}'.format(ks))
        
        #ix_user_jobs = numpy.array([not ('.batch' in s or '.extern' in s) for s in data['JobID']])
        N = len(numpy.unique(data['JobID_parent']) )
        jobs_summary = numpy.zeros( (N, ), dtype=data.dtype)
        #
        # ks should be a list of indices between 0 and len+1. This process still needs a little bit of work
        #  to guarantee subsets are not too long/short, etc. but this much should work
        #
        n_cpu = max(1, min(n_cpu, len(ks)-1))
        #
        # Traditional open(), close(), join() syntax here. This gives the same performance as the context handler syntax, though the cpu monitor behaves
        #  very differently -- at least on a Mac. The context manager seems to confuse OS-X perception of user vs system use, and a couple other things too.
        #P = mpp.Pool(n_cpu)
        #R = [P.apply_async(calc_jobs_summary, kwds={'data':working_data[k1:k2], 'verbose':verbose, 'n_cpu':1}) for k1,k2 in zip(ks[0:-1], ks[1:]) ]
        #P.close()
        #P.join()
        #results = [r.get() for r in R]
        #
        with mpp.Pool(n_cpu) as P:
            R = [P.apply_async(calc_jobs_summary, kwds={'data':working_data[k1:k2], 'verbose':verbose, 'n_cpu':1}) for k1,k2 in zip(ks[0:-1], ks[1:]) ]
            #
            results = [r.get() for r in R]
            if verbose:
                print('*** calc_jobs_summary: results [r.get() ] list set.')
        k=0
        for xx in results:
            #xx = r.get()
            dk=len(xx)
            #
            jobs_summary[k:k+dk]=xx
            k += dk
            #
            del xx
                    
            #
        #
        return jobs_summary
    #
    # else:
    # Single-process code:
    # this actually works quite differently. Not sure yet if it's a bad idea. Also, not clear if we want
    #  to be distinguishing jobs on JobID or JobID_parent.
    #
    #ix_user_jobs = numpy.array([not ('.batch' in s or '.extern' in s) for s in data['JobID']])
    # how do we do a distributed string operation in a numpy.array()?
    #
    # It looks like this was an expensive way to compute data['JobID_parent'], before we knew about _parent. This should save us some compute time!
    #job_ID_index = {ss:[] for ss in numpy.unique([s[0:(s+'.').index('.')]
    #                                            for s in data['JobID'][ix_user_jobs] ])}
    job_ID_index = {ss:[] for ss in numpy.unique( data['JobID_parent'] )}
    #
    jobs_summary = numpy.zeros( shape=(len(job_ID_index), ), dtype=data.dtype)
    #
    # build an index of unique ids : {jobid_parent:[k1, k2, k3...],}
    for k,s in enumerate(data['JobID_parent']):
        job_ID_index[s] += [k]
    #
    if verbose:
        print('Starting jobs_summary...')
    # we should be able to MPP this as well...
    #jobs_summary = numpy.array(numpy.zeros(shape=(len(job_ID_index), )), dtype=data.dtype)
    t_now = mpd.date2num(dtm.datetime.now())
    for k, (j_id, ks) in enumerate(job_ID_index.items()):
        if len(ks)==0:
            print('*** no ks: {}, {}'.format(j_id, ks))
        #
        # assign place-holder values to jobs_summary (we can probably skip this step)
        jobs_summary[k]=data[numpy.min(ks)]  # NOTE: these should be sorted, so we could use ks[0]
        #
        # NOTE: for performance, because sub-sequences should be sorted (by index), we should be able to
        #  use their index position, rather than min()/max()... but we'd need to be more careful about
        #. that sorting step, and upstream dependencies are dangerous...
        # NOTE: might be faster to index eact record, aka, skip sub_data and for each just,
        #. {stuff} = max(data['End'][ks])
        #
        # use a buffer
        sub_data = data[sorted(ks)]
        #
#        jobs_summary[k]['End'] = numpy.nanmax(sub_data['End'])
#        jobs_summary[k]['Start'] = numpy.nanmin(sub_data['Start'])
#        jobs_summary[k]['NCPUS'] = numpy.nanmax(sub_data['NCPUS'])
#        jobs_summary[k]['NNodes'] = numpy.nanmax(sub_data['NNodes']).astype(int)
#        jobs_summary[k]['NTasks'] = numpy.nanmax(sub_data['NTasks']).astype(int)
        #
        # This is the proper syntax (not forgiving) to assign a row:
        # again, was there a reason to NOT use nanmax()? Might be better to make our own nanmax(), like:
        # x = numpy.max(X[numpy.isnan(X).invert()])
        jobs_summary[['End', 'Start', 'NCPUS', 'NNodes', 'NTasks']][k] = numpy.nanmax(sub_data['End']),\
            numpy.nanmin(sub_data['Start']),\
            numpy.nanmax(sub_data['NCPUS']).astype(int),\
            numpy.nanmax(sub_data['NNodes']).astype(int),\
            numpy.nanmax(sub_data['NTasks']).astype(int)
            
        #
        # move this into the loop. weird things can happen with buffers...
        del sub_data
    #
    if verbose:
        print('** DEBUG: jobs_summary.shape = {}'.format(jobs_summary.shape))
    #
    return jobs_summary
#
def get_cpu_hours(n_points=10000, bin_size=7., t_min=None, t_max=None, jobs_summary=None, verbose=False, n_cpu=None, step_size=10000, d_t=None):
    '''
    # Loop-Loop version of get_cpu_hours. should be more memory efficient, might actually be faster by eliminating
    #  intermediat/transient arrays.
    #
    # Get total CPU hours in bin-intervals. Note these can be running bins (which will cost us a bit computationally,
    #. but it should be manageable).
    # NOTE: By permitting jobs_summary to be passed as a param, we make it easier to do a recursive mpp operation
    #. (if n_cpu>1: {split jobs_summary into n_cpu pieces, pass back to the calling function with n_cpu=1
    #
    # Some unfortunate naming conventions here will require clarification.
    # @n_points: as it sounds -- length of timeseries output; can be overridden by specifying d_t
    # @bin_size: aggregation bin size (eg, daily, weekly binning), but points in timeseries can overlap.
    # @t_min, @t_max: min,max times in timeseries. will automagically determine these from input data; specifying will override.
    # @jobs_summary: input jobs_summary data array
    # @verbose: verbose. probably boolean, but give an integer a try!
    # @n_cpu: n cpus for parallel processing. Currently thread based, but likely to switch to using multiprocess, which permits multi-node
    # @step_size: multiprocessing  chunks.
    # @d_t: time-step size. will override n_points.
    '''
    #
    # stash a copy of input prams:
    inputs = {ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')}
    n_cpu = (n_cpu or 1)
    cpuh_dtype = [('time', '>f8'),
        ('t_start', '>f8'),
        ('cpu_hours', '>f8'),
        ('N_jobs', '>f8')]
    #
    # use IX input to get a subset of the data, if desired. Note that this indroduces a mask (or something)
    #. and at lest in some cases, array columns should be treated as 2D objects, so to access element k,
    #. ARY[col][k] --> (ARY[col][0])[k]
    if jobs_summary is None:
        raise Exception('jobs_summary data not provided.')
        #
    if len(jobs_summary)==0:
        return numpy.array([], dtype=cpuh_dtype)
    #
    if verbose:
        print('** DEBUG: len(jobs_summary): {}'.format(len(jobs_summary)))
    #
    t_now = numpy.max([jobs_summary['Start'], jobs_summary['End']])
    #print('*** shape(t_now): ', numpy.shape(t_now))
    if verbose:
        print('** DEBUG: len(jobs_summary[ix]): {}'.format(len(jobs_summary)))
    #
    #
    t_start = jobs_summary['Start']
    t_end = jobs_summary['End'].copy()
    #
    if verbose:
        print('** DEBUG: (get_cpu_hours) initial shapes:: ', t_end.shape, t_start.shape, jobs_summary['End'].shape )
    #
    # handle currently running jobs (do we need to copy() the data?)
    t_end[numpy.logical_or(t_end is None, numpy.isnan(t_end))] = t_now
    #
    if verbose:
        print('** DEBUG: (get_cpu_hours)', t_end.shape, t_start.shape)
    #
    if t_min is None:
        t_min = numpy.nanmin([t_start, t_end])
    #
    if t_max is None:
        t_max = numpy.nanmax([t_start, t_end])
    #
    inputs.update({'t_min':t_min, 't_max':t_max})
    #
    # output container:
#    # TODO: consider (??) allowing d_t designation, instead of n_points.
#    #  this will better allow us to do complex masks, like M-F,8-5, etc.
#    # eg:
    if not d_t is None:
        # NOTE: this assumes t=0 bins (eg, d_t = .1, t_min=1.05 will give 1.0, 1.1, 1.2, ..., no 1.05, 1.15, ....
        #  this is (presently) by design, but it may be desirable to add a phase variable at some point, eg: ( (t_min//d_t)*d_t - dt_0 )
        ts = numpy.arange( (t_min//d_t)*d_t, (d_t + (t_max//d_t)*d_t), d_t )
        n_points = len(ts)
        #
    else:
        ts = numpy.linspace(t_min, t_max, n_points)
    #
    CPU_H = numpy.zeros( (n_points, ), dtype=cpuh_dtype)
    # NOTE: the ts[:] syntax is probably not necessary, but sometimes we have to be extra careful about by_val/by_reff assignment.
    #  we need to copy the values into the array; the exact behavior and syntax might vary between versions of Python and numpy.
    CPU_H['time'] = ts[:]
    CPU_H['t_start'] = CPU_H['time']-bin_size
    del ts
    #
    #CPU_H = numpy.zeros( (n_points, ), dtype=cpuh_dtype)
    #CPU_H['time'] = numpy.linspace(t_min, t_max, n_points)
    #CPU_H['t_start'] = CPU_H['time']-bin_size
    #
    # recursive MPP handler:
    #  this one is a little bit complicated by the use of linspace(a,b,n), sice linspae is inclusive for both a,b
    #  so we have to do a trick to avoid douple-calculationg (copying) the intersections (b_k-1 = a_k)
    if n_cpu>1:
        #
        # divide up jobs_summary to a bunch of processes
        # TODO: do this with a linspace().astype(int)
        ks = numpy.array([*numpy.arange(0, len(jobs_summary), step_size), len(jobs_summary)])
        #
        #
        with mpp.Pool(n_cpu) as P:
            R = []
            #for k_p, (t1, t2) in enumerate(zip(t_intervals[0:-1], t_intervals[1:])):
            for k1, k2 in zip(ks[0:-1], ks[1:]):
                #
                my_inputs = inputs.copy()
                my_inputs.update({'jobs_summary':jobs_summary[k1:k2], 'n_cpu':1})
                if verbose:
                    print('*** my_inputs: ', my_inputs)
                #
                R += [P.apply_async(get_cpu_hours, kwds=my_inputs)]
                #
            #
            results = [r.get() for r in R]
            for r in results:
                #CPU_H[['cpu_hours', 'N_jobs']][:] += r.get()[['cpu_hours', 'N_jobs']][:]
                #CPU_H['cpu_hours'] += r.get()['cpu_hours']
                #CPU_H['N_jobs'] += r.get()['N_jobs']
                CPU_H['cpu_hours'] += r['cpu_hours']
                CPU_H['N_jobs']    += r['N_jobs']
            #
            return CPU_H
    #
    # SPP code:
    #
    if verbose:
        print('*** ', cpu_h.shape)
    #
    for k,t in enumerate(CPU_H['time']):
        # TODO: wrap this into a (@jit compiled) function to parallelize? or add a recursive block
        #  to parralize the whole function, using an index to break it up.
        # NOTE / TODO: we *could* add additional masks or indices here to, for example, filter out weekends and after-hours (focus on M-F,8-5)
        #   jobs... but it's still hard
        ix_k = numpy.where(numpy.logical_and(t_start<=t, t_end>(t-bin_size) ))[0]
        #print('*** shape(ix_k): {}//{}'.format(ix_k.shape, numpy.sum(ix_k)) )
        #
        N_ix = len(ix_k)
        if N_ix==0:
            CPU_H[k] = t, t-bin_size,0.,0.
            continue
        #
        #print('** ** ', ix_k)
        #CPU_H[k] = t, t-bin_size, numpy.sum( numpy.min([t*numpy.ones(N_ix), t_end[ix_k]], axis=0) -
        #                    numpy.max([(t-bin_size)*numpy.ones(N_ix), t_start[ix_k]]) )*24., N_ix
        #print('*** *** ', k,t, N_ix, ix_k)
        #
        # TODO: this can be made faster (probably?) by basically counting all the middle elements and only dong max(), min(), and arithmetic
        #  on the leading and trailing elemnents. But if that breaks vectorization, we'll give up those gains.
        CPU_H[['cpu_hours', 'N_jobs']][k] = numpy.sum( (numpy.min([t*numpy.ones(N_ix), t_end[ix_k]], axis=0) -
            numpy.max( [(t-bin_size)*numpy.ones(N_ix), t_start[ix_k]], axis=0))*24.*(jobs_summary['NCPUS'])[ix_k] ), N_ix
    #
    #CPU_H['cpu_hours']*=24.
    #print('*** returning CPU_H:: ', numpy.shape(CPU_H))
    return CPU_H
#
#def active_jobs_cpu(n_points=5000, ix=None, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, jobs_summary=None, verbose=None, mpp_chunksize=None):
def active_jobs_cpu(n_points=5000, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, jobs_summary=None, verbose=None, mpp_chunksize=10000):
    '''
    # Since this is now procedural, let's phase out ix?. the class-resident version can keep it if so desired.
    # NOTE on parallelization: There is quite a bit of pre-processing front-matter, so it makes sense to separate the SPP and MPP, as opposed
    #  to doing a recursive callback. Basically, we put in a fair bit of work to construct our working data; it would be expensive to do it again,
    #  messy to integrate it into the call signature, etc. In other words, if you're adding a lot of inputs and input handling to accomodate MPP,
    #  it's probably better to just outsource it. See MPP section below.
    #
    # TODO: assess parallel strategy; maybe adjust chunksize to avoid too-big-to-parallel problems.
    # @n_points: number of points in returned time series.
    # @bin_size: size of bins. This will override n_points
    # @t_min: start time (aka, bin phase).
    # @ix: an index, aka user=my_user
    # @mpp_chunksize: batch size for MPP, in this case probably an apply_async(). Note that this has a significant affect on speed and parallelization
    #  performance, and optimal performance is probably related to maximal use of cache. For example, for moderate size data sets, the SPP speed might be ~20 sec,
    #  and the 2cpu, chunk_size=20000 might be 2 sec, then 1.5, 1.2, for 3,4 cores. toying with the chunk_size parameter suggests that cache size is the thing. It is
    #  probably worth circling back to see if map_async() can be made to work, since it probably has algorithms to estimate optimal chunk_size.
    '''
    #
    # TODO: modify this to allow a d_t designation, instead of n_points, so we can specify -- for example, an hourly sequence.
    # DEBUG: for now, just set n_cpu=1. we should be able to run this and fix the mpp later...
    #n_cpu=1
    #
    # empty container:
    null_return = lambda: numpy.array([], dtype=[('time', '>f8'),('N_jobs', '>f8'),('N_cpu', '>f8')])
    if verbose is None:
        verbose = 0
    #
    mpp_chunksize = mpp_chunksize or 1000
    n_cpu = n_cpu or 1
    #
    if jobs_summary is None or len(jobs_summary) == 0:
        return null_return()
    #
    if t_now is None:
        t_now = numpy.max([jobs_summary['Start'], jobs_summary['End']])
    #
    if len(jobs_summary)==0:
        return null_return()
    #
    t_start = jobs_summary['Start']
    t_end = jobs_summary['End']
    t_end[numpy.logical_or(t_end is None, numpy.isnan(t_end))] = t_now
    #
    #print('** DEBUG: ', t_end.shape, t_start.shape)
    #
    if t_min is None:
        t_min = numpy.nanmin([t_start, t_end])
    #
    if verbose:
        print('** DEBUG: shapes:: {}, {}'.format(numpy.shape(t_start), numpy.shape(t_end)))
    #
    if len(t_start)==1:
        print('** ** **: t_start, t_end: ', t_start, t_end)
    #
    # catch an empty set:
    # OR or AND condition???
    if len(t_start)==0 or len(t_end)==0:
        print('Exception: no data:: {}, {}'.format(numpy.shape(t_start), numpy.shape(t_end)))
        return null_return()
    #
    if t_max is None:
        t_max = numpy.nanmax([t_start, t_end])
    #
    if not (bin_size is None or t_min is None or t_max is None):
        n_points = int(numpy.ceil(t_max-t_min)/bin_size)
    print(f'*** DEBUG: {n_points}, {bin_size}')
    output = numpy.zeros( n_points, dtype=[('time', '>f8'),
                ('N_jobs', '>f8'),
                ('N_cpu', '>f8')])
    output['time'] = numpy.linspace(t_min, t_max - (t_max - t_min)/n_points, n_points)
    #
    # This is a slick, vectorized way to make an index, but It uses way too much memory, and so is probably too slow anyway
    #IX_t = numpy.logical_and( X.reshape(-1,1)>=t_start, (X-bin_size).reshape(-1,1)<t_end )
    #
    # Here's the blocked out loop-loop version. Vectorized was originally much faster, but maybe if we pre-set the array
    #
    if n_cpu==1:
#        for j, t in enumerate(output['time']):
#            # numpy.where() might be pretty expensive, so only use if memory is a major issue
#            #
#            #ix_t = numpy.where(numpy.logical_and(t_start<=t, t_end>t))[0]
#            #output[['N_jobs', 'N_cpu']][j] = len(ix_t), numpy.sum(jobs_summary['NCPUS'][ix_t])
#            #
#            # but this is still looping on t, where for MPP we've gotten away from that. maybe just call process_ajc_rows() ?
#            ix_t = numpy.logical_and(t_start<=t, t_end>t)
#            output[['N_jobs', 'N_cpu']][j] = numpy.sum(ix_t), numpy.sum(jobs_summary['NCPUS'][ix_t])
#            #
#            if verbose and j%1000==0:
#                print('*** PROGRESS: {}/{}'.format(j,len(output)))
#                print('*** ', output[j-10:j])
#        #return output
        #r_nj, r_ncpu = process_ajc_rows(t=output['time'], t_start=t_start, t_end=t_end, NCPUs=jobs_summary['NCPUS'])
        #output['N_jobs'] = r_nj
        #output['N_cpu']  = r_ncpu
        #
        # can we save some time by consolidating this into a single statement?
        # NOTE: Check this... using all of t (might?) produces an artifact where active_cpus -> 0 at the end, because that
        #   time ends up falls outside the start/stop time domain??? I guess though... since the time axis is constructecd
        #   separately, just excluding the last element is not a very good fix; we should live with the artifact or fix the
        #   time axis.
        output['N_jobs'], output['N_cpu'] = process_ajc_rows(t=output['time'], t_start=t_start, t_end=t_end, NCPUs=jobs_summary['NCPUS'])
        return output
        #
        #return numpy.core.records.fromarrays([numpy.linspace(t_min, t_max, n_points), *process_ajc_rows(t=numpy.linspace(t_min, t_max, n_points), t_start=t_start, t_end=t_end, NCPUs=jobs_summary['NCPUS'])], dtype=[('time', '>f8'),
        #        ('N_jobs', '>f8'), ('N_cpu', '>f8')])
    else:
#        output = numpy.zeros( n_points, dtype=[('time', '>f8'),
#                ('N_jobs', '>f8'),
#                ('N_cpu', '>f8')])
#        output['time'] = numpy.linspace(t_min, t_max, n_points)
        #
        n_procs = min(numpy.ceil(len(t_start)/n_cpu).astype(int), numpy.ceil(len(t_start)/mpp_chunksize).astype(int) )
        ks_r = numpy.linspace(0, len(t_start), n_procs+1).astype(int)
        #
        # NOTE: use smaller of n_cpu, n_procs actual processing units.
        with mpp.Pool( min(n_cpu, n_procs) ) as P:
            # need to break this into small enough chunks to not break the pickle() indexing.
            # smaller of N/n_cpu or N/chunk_size
            #
            results = [P.apply_async(process_ajc_rows,  kwds={'t':output['time'], 't_start':t_start[k1:k2],
                        't_end':t_end[k1:k2], 'NCPUs':jobs_summary['NCPUS'][k1:k2]} ) for k1, k2 in zip(ks_r[:-1], ks_r[1:])]
            #
            R = [r.get() for r in results]
        #  still can't quite explain why the SPP seems to be much much much (non-linerly) slower than mpp (on Mazama)
        # TODO: how do we numpy.sum(R) on the proper axis?
        for k_r, (r_nj, r_ncpu) in enumerate(R):
            #print('**** [{}]: {} ** {}'.format(k_r, r_nj[0:10], r_ncpu[0:10]))
            # TODO: There is a syntax to do this in one statement. Also, is there an advantage (or problem with async calculation) to do
            # ... but I can't seem to find a working syntax for something like [structured_array_cols] = [non-structured_array_cols] for
            #  multiple columns. So we could modify process_ajc_rows() to return a structured array, or we can just do this.
            #
            if verbose:
              print('*** DEBUG: shape(output)={}, shape(r_nj)={}, shape(r_ncpu):{}'.format(numpy.shape(output), numpy.shape(r_nj), numpy.shape(r_ncpu)))
            output['N_jobs'] += numpy.array(r_nj)
            output['N_cpu']  += numpy.array(r_ncpu)
        #
        del P, R, results
        #
    return output
#
def process_ajc_rows(t, t_start, t_end, NCPUs):
    # active jobs cpus...
    #
    #print('*** DEBUG: {} ** {}'.format(len(t_start), len(t_end)))
    # Because we artifically set t_end = t_now for active jobs, I think using t_end>t, vs t_end >= t
    # produces an artifact of usage falling off. We don't have to use the exclusive
    # t1 <= t2 < t or t1 < t <= t2 restriction since we're not counting t; we're just asking if a job is
    # active during a time bin. note we could also set t_end = now()+dt.
    #ix_t = numpy.logical_and(t_start<=t.reshape(-1,1), t_end>t.reshape(-1,1))
    ix_t = numpy.logical_and(t_start <= t.reshape(-1,1), t_end >= t.reshape(-1,1))
    #
    return numpy.array([numpy.sum(ix_t, axis=1), numpy.array([numpy.sum(NCPUs[js]) for js in ix_t])])
#
def running_mean(X,n=10):
    return (numpy.cumsum(numpy.insert(X,0,0))[n:] - numpy.cumsum(numpy.insert(X,0,0))[:-n])/n


