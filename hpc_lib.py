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
#import numba
import pandas
#
# TODO: so... do we keep our plotting routines separate, or do we just make sure we use... uhh? (double check this)
# matplotlib.use('Agg')
#. load this first, so on an HPC primary 
import pylab as plt
#
day_2_sec=24.*3600.
#
# NOTE: matplotlib.dates changed the standard reference epoch. This is mainly to mitigate roundin errors in modern
#  dates. You can use set_/get_epoch() to read and modify the epoch, but there are rules about doing this
#  before figures are plotted, etc. It's typical affect is cosmetic, so differencing dates, elapsed times, etc.
#  will not change (much). Looks like the best way to handle this is to augment the numerical dates.
old_mpd_epoch = '0000-12-31T00:00:00'
new_mpd_epoch = '1970-01-01T00:00:00'
dt_mpd_epoch = 719163.0
#
# TODO: move group_ids, and local-specific things like that to site-specific  modules or data files.
#
# GIT NOTE: Mazama is gone; no need for mazama_groups_ids
#
# TODO: LOTS more serc_user ids!
# Generally, I think we don't use these any longer. We get these natively from the data.
#serc_user_ids = ['biondo', 'beroza', 'sklemp', 'harrisgp', 'gorelick', 'edunham', 'sagraham', 'omramom', 'aditis2', 'oneillm', 'jcaers', 'mukerji', 'glucia', 'tchelepi', 'lou', 'segall', 'horne', 'leift']

# These are used for some reporting code, that should be moved
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
def elapsed_time_2_day(tm_in, verbose=0):
    #
    if tm_in in ( 'Partition_Limit', 'UNLIMITED' ) or tm_in is None:
    #if tm_in.lower() in ( 'partition_limit', 'unlimited' ):
        return None
    #
    return elapsed_time_2_sec(tm_in=tm_in, verbose=verbose)/(day_2_sec)
#
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
kmg_vals = {'k':1E3, 'm':1E6, 'g':1E9, 't':1E12}
def kmg_to_num(x):
    '''
    # convert something like 25k to 25000
    # for now, assume xxxU format.
    '''
    #
    #if isinstance(x, (float, int)):
    #    return x
    try:
        return float(x)
    except:
      pass
    #
    if x is None or x=='':
      return None
    #
    try:
        return float(x[:-1]) * kmg_vals[x[-1].lower()]
    except:
        return None
def space_replacer(s, space_to='_', space_ord=32):
    # NOTE: default is to replace space with '_', but can be abstracted. Let's also handle bytearrays...
    # NOTE: we'll put in a hack for bytes array, but we should be getting the ecoding from the input.
    # for now, let's not do this. bytes vs string can be handled by the calling function. If we handle it here, we
    # restrict it to a sort of hacked handling.
    #if isinstance(s,bytes):
    #    return s.replace(chr(space_ord).encode(), space_to.encode())
    #
    #if not isinstance(s,str):
    #    s = str(s)
    return str(s).replace(chr(space_ord), space_to)
#
#
dtm_handler_default = str2date_num
# TODO: write this as a class, inherit from dict, class(dict):
default_SLURM_types_dict = {'User':str, 'JobID':str, 'JobName':str, 'Partition':str, 'State':str, 'JobID_parent':str,
        'Timelimit':elapsed_time_2_day,
            'Start':dtm_handler_default, 'End':dtm_handler_default, 'Submit':dtm_handler_default,
                    'Eligible':dtm_handler_default,
                'Elapsed':elapsed_time_2_day, 'MaxRSS':kmg_to_num, 'AveRSS':kmg_to_num, 'ReqMem':kmg_to_num,
                'MaxVMSize':kmg_to_num,'AveVMSize':kmg_to_num,  'NNodes':int, 'NCPUS':int,
                 'MinCPU':str, 'SystemCPU':elapsed_time_2_day, 'UserCPU':elapsed_time_2_day, 'TotalCPU':elapsed_time_2_day,
                'NTasks':int,'MaxDiskWrite':kmg_to_num, 'AveDiskWrite':kmg_to_num,
                    'MaxDiskRead':kmg_to_num, 'AveDiskRead':kmg_to_num
                }
#
# yoder, 2022-08-11: updating default_SLURM_types_dict, mostly to facilitate SQUEUE calls. separating it out
#   (rather than just directly add in the new cols) for posterity. also, adding .lower() and .upper() duplicate
#   entries to simplify searching.
default_SLURM_types_dict.update({ky:int for ky in ['NODES', 'CPUS', 'TASKS', 'numnodes', 'numtasks', 'numcpus']})
default_SLURM_types_dict.update({'TIMELEFT':elapsed_time_2_day, 'TIMEUSED':elapsed_time_2_day})
dst_ul = {ky.lower():val for ky,val in default_SLURM_types_dict.items()}
dst_ul.update( {ky.upper():val for ky,val in default_SLURM_types_dict.items()} )
default_SLURM_types_dict.update(dst_ul)
del dst_ul
# TODO: this addition to default_SLURM_types_dict separated for diagnistic/dev purposes. Integrate it directly into main def.
default_SLURM_types_dict.update({'Account':str})
#
def running_mean(X, n=10):
    return (numpy.cumsum(X)[n:] - numpy.cumsum(X)[:-n])/n
#
class SACCT_data_handler(object):
    #
    # TODO: write GRES (gpu, etc. ) handler functions.
    # TODO: add real-time options? to get sacct data in real time; we can still stash it into a file if we need to, but
    #   more likely a varriation on load_sacct_data() that executes the sacct querries, instead of loading a file.
    dtm_handler_default = str2date_num
    #
    default_types_dict = default_SLURM_types_dict
    #
    time_units_labels={'hour':'hours', 'hours':'hours', 'hr':'hours', 'hrs':'hours', 'min':'minutes','minute':'minutes', 'minutes':'minutes', 'sec':'seconds', 'secs':'seconds', 'second':'seconds', 'seconds':'seconds'}
    #    
    time_units_vals={'days':1., 'hours':24., 'minutes':24.*60., 'seconds':24.*3600.}
    #
    # add a dict to capture input(like) paremeters. These can then be easily added to hdf5 (and similar) storage
    #  containers.
    attributes={}
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
        n_cpu = int(n_cpu or 1)
        n_cpu = int(n_cpu)
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
        self.attributes.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
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
        self.attributes['h5out_file'] = h5out_file
        self.attributes['RH'] = self.RH
        #
        # NOTE: matplotlib.dates changed its default reference epoch, for numerical date conversion, from
        #   0000-12-31 to 1970-1-1. switching the epoch can be tricky; different versions of mpd can give
        #   weird years. let's handle it here, assuming modern dates:
#        dt_epoch = 0.
#        yr_test = mpd.num2date(SACCT_obj.jobs_summary['Start'][0]).year
#        if yr_test > 3000:
#            dt_epoch = -dt_mpd_epoch
#        if yr_tesst < 1000:
#            dt_epoch =  dt_mod_epochj
        # what does this do? I think we can get rid of it...
        #dt_mod_epoch = self.compute_mpd_epoch
        #
        self.dt_mpd_epoch = self.compute_mpd_epoch_dt()
    #
    def __getitem__(self, *args, **kwargs):
        return self.jobs_summary.__getitem__(*args, **kwargs)
    def __setitem__(self, *args, **kwargs):
        return self.jobs_summary.__setitem__(*args, **kwargs)
    #
    @property
    def dtype(self):
        if 'jobs_summary' in self.__dict__.keys():
            return self.jobs_summary.dtype
        else:
            return None
    #
    def get_NGPUs(self, jobs_summary=None, alloc_tres=None):
        if alloc_tres is None:
            if jobs_summary is None:
                jobs_summary = self.jobs_summary
            alloc_tres = jobs_summary['AllocTRES'].astype(str)
        #
        else:
            alloc_tres = numpy.array(alloc_tres).astype(str)
        #
        #return numpy.array([float(s.split('gpu=')[1].split(',')[0]) if 'gpu=' in s else 0.
        # for s in jobs_summary['AllocTRES'].astype(str)])
        #
        # NOTE: if this blows up, it might be because I made a couple minor type-handling changes on 25 July 2022 (yoder):
        return numpy.array([int(s.split('gpu=')[1].split(',')[0]) if 'gpu=' in s else 0
         for s in alloc_tres]).astype(int)
    #
    def compute_mpd_epoch_dt(self, test_col='Submit', test_index=0, yr_upper=3000, yr_lower=1000):
        #
        #print('*** DEBUG: ', len(self.jobs_summary), len(self[test_col]))
        #print('*** DEBUG: ', self[test_col][0:10])
        test_date = None
        while test_date is None:
            x = self[test_col][test_index]
            if not isinstance(x,float) or isinstance(x,int) or isinstance(x,dtm.datetime):
                test_index += 1
                continue
            test_date = self.jobs_summary[test_col][test_index]
        if test_date is None:
            test_date = dtm.datetime.now(pytz.timezone('UTC'))
            #
        #
        if isinstance(test_date, dtm.datetime):
            test_date = mpd.date2num(test_date)
#        dt_epoch = 0.
#        yr_test = mpd.num2date(self.jobs_summary[test_col][test_index]).year
#        if yr_test > yr_upper:
#            dt_epoch = -dt_mpd_epoch
#        if yr_test < yr_lower:
#            dt_epoch =  dt_mod_epoch
        #
        #return dt_epoch

        return compute_mpd_epoch_dt(test_date, yr_upper=yr_upper, yr_lower=yr_lower)
    #
    def calc_summaries(self, n_cpu=None, chunk_size=None, t_min=None, t_max=None):
        #
        n_cpu = n_cpu or self.n_cpu
        n_points_usage = self.n_points_usage
        chunk_size = chunk_size or self.chunk_size
        #
        # TODO: add start_date, end_date to parent class. Also then change all reffs of t_min/_max to start_date,
        #   end_date to simplify the syntax.
        # add t_min, t_max (start/end dates). This is a little bit clunky and bass-ackwards, since it is actually
        #  facilitating compatibility with chile subclasses, but in a way that is harmless. Also, might be a good idea to
        #  handle t_min, t_max in general. These are the min/max times in the timeseries. By default, they are inferred from
        #  the data, but because SACCT yields all jobs in any state during the start/end times, you can actually get start/end
        #  times outside your query window.
        if t_min is None and 'start_date' in self.__dict__.keys():
            t_min = self.start_date
            #
        if t_max is None and 'end_date' in self.__dict__.keys():
            t_max = self.end_date
            #
        #
        # t_min, t_max need to be numeric. most likely, we'll be converting
        #  from a datetime type, but this should be better handled.
        # TODO: handle data type better...
        if not (isinstance(t_min, float) or isinstance(t_min, int)):
            t_min = mpd.date2num(t_min)
        if not (isinstance(t_max, float) or isinstance(t_max, int)):
            t_max = mpd.date2num(t_max)
        #
        self.jobs_summary = self.calc_jobs_summary()
        if not self.keep_raw_data:
            del self.data
        #
        self.cpu_usage = self.active_jobs_cpu(n_cpu=n_cpu, t_min=t_min, t_max=t_max, mpp_chunksize=min(int(len(self.jobs_summary)/n_cpu), chunk_size) )
        self.weekly_hours = self.get_cpu_hours(bin_size=7, n_points=n_points_usage, t_min=t_min, t_max=t_max, n_cpu=n_cpu)
        self.daily_hours = self.get_cpu_hours(bin_size=1, n_points=n_points_usage, t_min=t_min, t_max=t_max, n_cpu=n_cpu)
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
    def write_hdf5(self, h5out_file=None, append=False, meta_data=None):
        h5out_file = h5out_file or self.h5out_file
        if h5out_file is None:
            h5out_file = 'sacct_writehdf5_output.h5'
        #
        h5_mode = 'a'
        if not append:
            h5_mode = 'w'
            with h5py.File(h5out_file, 'w') as fout:
                # over-write this file
                pass
        #
        if 'data' in self.__dict__.keys():
            array_to_hdf5_dataset(input_array=self.data, dataset_name='data', output_fname=h5out_file, h5_mode='a')
        #
        # TODO: consider that we really don't need the computed data sets any longer. computing cpu_Usage, etc. is just not that hard to do. not hurting anything
        #  but not necessary either.
        array_to_hdf5_dataset(input_array=self.jobs_summary, dataset_name='jobs_summary', output_fname=h5out_file, h5_mode='a')
        array_to_hdf5_dataset(input_array=self.cpu_usage, dataset_name='cpu_usage', output_fname=h5out_file, h5_mode='a')
        array_to_hdf5_dataset(input_array=self.weekly_hours, dataset_name='weekly_hours', output_fname=h5out_file, h5_mode='a')
        array_to_hdf5_dataset(input_array=self.daily_hours, dataset_name='daily_hours', output_fname=h5out_file, h5_mode='a')
        #
#        # TODO: also write metadata/attributes:
#        # you'd think the best thing to do would be to make a datset for inptus, but then we have to structure
#          that dataset. I think we can just store variables.
        with h5py.File(h5out_file, 'a') as fout:
            #for ky in ('start_date', 'end_date'):
            #    fout.attrs.create(ky, simple_date_string(self.__dict__[ky]))
            #
            for ky, f in [('partition', str), ('group',str), ('start_date', simple_date_string), ('end_date', simple_date_string)]:
                #fout.attrs.create(ky, f(self.__dict__.get(ky,None)))
                if ky in self.__dict__.keys():
                    fout.attrs.create(ky, f(self.__dict__[ky]))
        
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
        n_cpu = int(n_cpu)
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
    def process_row(self, rw, headers=None, RH=None, delim=None):
        # use this with MPP processing:
        # ... but TODO: it looks like this is 1) inefficient and 2) breaks with large data inputs because I think it pickles the entire
        #  class object... so we need to move the MPP object out of class.
        #
        # get rid of spaces. Values like State="CANCELED by 12345" seem to generate errors under some
        #   circumstances?
        rw = rw.replace(chr(32), '_')
        #
        headers = (headers or self.headers)
        RH      = (RH or self.RH)
        if delim is None:
            delim = self.delim
        # use this for MPP processing:
        rws = rw.split(delim)
        #print('*** DEBUG lens: ', len(rws), len(headers))
        #return [None if vl=='' else self.types_dict.get(col,str)(vl)
        #            for k,(col,vl) in enumerate(zip(self.headers, rw.split(self.delim)[:-1]))]
        # NOTE: last entry in the row is JobID_parent.
        #
        # DEBUG:
#        print('*** rw: ', rw)
#        print('*** hdrs: ', headers)
#        print('*** RH: ', RH)
        #
        try:
            return [None if vl=='' else self.types_dict.get(col,str)(vl)
                    for k,(col,vl) in enumerate(zip(headers, rws[:-1]))] + [rws[RH['JobID']].split('.')[0]]
        except:
            print('*** EXCEPTION! with row: ', rw)
            print('*** HEADERS: ', headers)
            print('*** RH: ', RH)
            raise Exception()
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
    # GIT NOTE: deleting get_cpu_hours_depricated()
    #  Current commit: commit d0872dbf00fd493fa4937d4b9030f9b5a927e21d
    #
    def active_jobs_cpu(self, n_points=5000, ix=None, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, jobs_summary=None, verbose=None, mpp_chunksize=None, nan_to=0., NCPUs=None):
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
        #
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
            verbose=verbose, mpp_chunksize=mpp_chunksize, nan_to=nan_to, NCPUs=NCPUs)
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
        ix = self.jobs_summary['NCPUS']==n_cpu
        return self.jobs_summary['Start'][ix] - self.jobs_summary['Submit'][ix]
    #
    def get_active_cpus_layer_cake(self, jobs_summary=None, layer_field='Partition', layers=None, n_points=5000, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, verbose=False, mpp_chunksize=10000, nan_to=0., NCPUs=None):
        # (n_points=5000, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, jobs_summary=None, verbose=None, mpp_chunksize=10000, nan_to=0.
        if jobs_summary is None:
            jobs_summary = self.jobs_summary
        if NCPUs is None or NCPUs=='':
            NCPUs = jobs_summary['NCPUS']
        #
        if t_now is None:
            t_now = mpd.date2num(dtm.datetime.now() )
        if t_min is None:
            t_min = numpy.nanmin([jobs_summary['Start'], jobs_summary['End']])
            #t_min = numpy.nanmin( [t_now, t_min] )
        #
        if t_max is None:
            t_max = numpy.nanmax([jobs_summary['Start'], jobs_summary['End']])
            print(f'*** DEBUG t_now: {t_now}, t_max: {t_max}')
            t_max = numpy.nanmin([t_now, t_max])
        #
        if layers is None:
            layers = [ky.decode() if hasattr(ky,'decode') else ky for ky in list(set(jobs_summary[layer_field]))]
        layers = {ky:{} for ky in layers}
        if verbose:
            print('*** ', layers)
        #
        dtype_cpuh = [(s, '>f8') for s in ['time'] + list(layers.keys())]
        dtype_jobs = dtype_cpuh
        #
        for j, ky in enumerate(layers.keys()):
            # TODO: better to handle b'' vs '' via ix.astype(str), or similar approach.
            ix = jobs_summary[layer_field] == (ky.encode() if hasattr(jobs_summary[layer_field][0], 'decode') else ky)
            XX = self.active_jobs_cpu(n_points=n_points, bin_size=bin_size, t_min=t_min, t_max=t_max, t_now=t_now, n_cpu=n_cpu,
                                     jobs_summary=jobs_summary[ix], verbose=verbose, mpp_chunksize=mpp_chunksize, nan_to=nan_to,
                                     NCPUs=NCPUs[ix])
            #
            if verbose:
                print('*** DEBUG: {}:: {}'.format(j, len(XX['N_cpu'])))
                #print(f'*** DEBUG: {k}:: {len(XX['N_cpu'])}')
            #
            # create output_ arrays for j==0 (or equivalently output_cpus and output_jobs do not exist...)
            if j==0:
                output_cpus = numpy.empty( (len(XX['N_cpu']), ), dtype=dtype_cpuh )
                output_jobs = numpy.empty( (len(XX['N_jobs']), ), dtype=dtype_jobs )
            #
            output_cpus[ky] = XX['N_cpu'][:]
            output_jobs[ky] = XX['N_jobs'][:]
        #
        output_cpus['time'] = XX['time']
        output_jobs['time'] = XX['time']
        #
        return {'N_cpu': output_cpus, 'N_jobs': output_jobs}
        
    #
    def get_cpu_hours_layer_cake(self, jobs_summary=None, layer_field='Partition', layers=None, bin_size=7, n_points=5000,
                              t_min=None, t_max=None, t_now=None, verbose=0):
        '''
        # revised version of gchlc with a more intuitive output.
        #
        '''
        #
        if jobs_summary is None:
            jobs_summary = self.jobs_summary
        #
        # t_min, t_max: These constraints need to be handled up-front, from the whole data set. Subsets will likely return
        #   different t_min/t_max.
        if t_now is None:
            t_now = mpd.date2num(dtm.datetime.now())
        if t_min is None:
            t_min = numpy.nanmin([jobs_summary['Start'], jobs_summary['End']])
            #t_min = numpy.nanmin([t_min, t_now])
        #
        if t_max is None:
            t_max = numpy.nanmax([jobs_summary['Start'], jobs_summary['End']])
            #
            # if data set contains unfinished jobs, set t_now to now. this probaby should be revisited... maybe makes more sense
            #  to use max(t_end) as the default t_now... Another point for this, it's a lot easier to set t_max=now() than
            #  max(in_data_set). Doing this can truncate active jobs, but will avoid artifacts.
            #if None in jobs_summary['End']:
            #    t_max = numpy.nanmax([t_max, t_now])
        #
        if layers is None:
            layers = [ky.decode() if hasattr(ky,'decode') else ky for ky in list(set(jobs_summary[layer_field]))]
        layers = {ky:{} for ky in layers}
        if verbose:
            print('*** ', layers)
        #
        dtype_cpuh = [(s, '>f8') for s in ['time'] + list(layers.keys())]
        dtype_jobs = dtype_cpuh
        
        output_cpuh = numpy.empty( (n_points, ), dtype=dtype_cpuh )
        output_jobs = numpy.empty( (n_points, ), dtype=dtype_jobs )
        output_elapsed = {s:0 for s in layers}
        #
        for j, ky in enumerate(layers.keys()):
            if verbose:
                print(f'*** ky: {ky} // {ky.encode()}')
            #
            # NOTE: should not need the en/decode() logic any more; that should be fixed in the HDF5 subclase, but it won't hurt...
            ix = jobs_summary[layer_field] == (ky.encode() if hasattr(jobs_summary[layer_field][0], 'decode') else ky)
            XX = self.get_cpu_hours(bin_size=bin_size, n_points=n_points, t_min=t_min, t_max=t_max,
                                                jobs_summary=jobs_summary[ix])
            output_cpuh[ky] = XX['cpu_hours'][:]
            output_jobs[ky] = XX['N_jobs'][:]
            output_elapsed[ky] =  numpy.sum(jobs_summary['Elapsed'][ix]*jobs_summary['NCPUS'][ix])
        #
        output_cpuh['time'] = XX['time']
        output_jobs['time'] = XX['time']
        #
        return {'cpu_hours': output_cpuh, 'jobs':output_jobs, 'elapsed':output_elapsed}
            
    # REMOVED:
    #def get_cpu_hours_layer_cake_depricated(self, jobs_summary=None, layer_field='Partition', layers=None, bin_size=7, n_points=5000,
#                              t_min=None, t_max=None, verbose=0):
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
    def report_activecpus_jobs_layercake_and_CDFs(self, group_by='Group', acpu_layer_cake=None, jobs_summary=None, ave_len_days=5., qs=[.5, .75, .9], n_points=5000, bin_size=None, fg=None, ax1=None, ax2=None, ax3=None, ax4=None):
        '''
        #  Inputs:
        #   @group_by: column name for grouping
        #   @acpu_jobs: optional input of cpuh_jobs
        #   @bin_size: bins size of timeseries.
        #   @fg: (optional) figure for plots. Should have 4 subplots. if not provided a default will be created.
        #   @ax1,2,3,4: Optional. if not provided will be created and/or drawn from fg.
        #   @ave_len_days: smoothing/averaging interval, in days.
        #   #qs: list/array-like of quantile thresholds.
        '''
        qs = numpy.array(qs)
        #bin_size = bin_size or .1
        if fg is None:
            fg = plt.figure(figsize=(20,16))
            for k in range(1,5):
                fg.add_subplot(2,2,k)
        #
        # NOTE: this might break if (fg is None), even if ax1,2,3,4 are provided)
        #ax1, ax2, ax3, ax4 = fg.get_axes()
        ax1 = ax1 or fg.get_axes()[0]
        ax2 = ax2 or fg.get_axes()[1]
        ax3 = ax3 or fg.get_axes()[2]
        ax4 = ax4 or fg.get_axes()[3]
        #
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        #
        ax1.set_title('Active CPUs', size=16)
        ax2.set_title('Active Jobs', size=16)
        ax3.set_title('Active CPUs, CDF', size=16)
        ax4.set_title('Active Jobs  CDF', size=16)
        #
        if acpu_layer_cake is None:
            acpu_layer_cake = self.get_active_cpus_layer_cake(layer_field=group_by, n_points=n_points, bin_size=bin_size, jobs_summary=jobs_summary)
        #
        #cpus = acpu_layer_cake['N_cpu']
        #jobs = acpu_layer_cake['N_jobs']
        T = acpu_layer_cake['N_cpu']['time']
        #
        lc_ncpu = plot_layer_cake(data=acpu_layer_cake['N_cpu'], ax=ax1)
        lc_njobs = plot_layer_cake(data=acpu_layer_cake['N_jobs'], ax=ax2)
        #
        # get an averaging length (number of TS elements) of about ave_len_days
        ave_len = int(numpy.ceil(ave_len_days*len(T)/(T[-1] - T[0])))
        z_cpu = ax1.get_lines()[-1].get_ydata()
        z_jobs = ax2.get_lines()[-1].get_ydata()
        z_cpus_smooth = running_mean(z_cpu, ave_len)
        z_jobs_smooth = running_mean(z_jobs, ave_len)
        #
        ax1.plot(T[-len(z_cpus_smooth):], z_cpus_smooth, ls='-', marker='', lw=2, label=f'{ave_len_days} days-ave')
        ax2.plot(T[-len(z_jobs_smooth):], z_jobs_smooth, ls='-', marker='', lw=2, label=f'{ave_len_days} days-ave')
        #
        qs_cpus = numpy.quantile(z_cpu, qs)
        qs_jobs = numpy.quantile(z_jobs, qs)
        #
        hh_cpus = ax3.hist(z_cpu, bins=100, cumulative=True, density=True, histtype='step', lw=3.)
        for x,y in zip(qs_cpus, qs):
            #ax3.plot([0., qs_cpus[-1], qs_cpus[-1]], [qs[-1], qs[-1], 0.], ls='--', color='r', lw=2. )
            ax3.plot([0., x, x], [y, y, 0.], ls='--', lw=2., label=f'{y*100.}th %: {x:.0f} cpus' )
        #
        hh_jobs = ax4.hist(z_jobs, bins=100, cumulative=True, density=True, histtype='step', lw=3.)
        for x,y in zip(qs_jobs, qs):
            #ax3.plot([0., qs_cpus[-1], qs_cpus[-1]], [qs[-1], qs[-1], 0.], ls='--', color='r', lw=2. )
            ax4.plot([0., x, x], [y, y, 0.], ls='--', lw=3., label=f'{y*100.}th %: {x:.0f} jobs' )
        #
        for ax in (ax1, ax2, ax3, ax4):
            ax.legend(loc=0)
        #
        return fg
        
    def report_cpuhours_jobs_layercake_and_pie(self, group_by='Group', cpuh_jobs=None, bin_size=.1, jobs_summary=None, wedgeprops=None, autopct=None, fg=None, ax1=None, ax2=None, ax3=None, ax4=None):
        '''
        # 2 x 2 block of figures showing CPU hours and active jobs, layer-cake by group_by and pie charts of same quantities.
        #  note that CPU hours are constructed timeseries (Either by creating a TS from each time interval and then mapping that to a
        #  master timeseries, or by looping over said master timeseries and computing how many bins fall inside the jobs_summary interval
        #  (which sounds like sort of a stupid way to compute that, but it does have some merits...).
        #  Inputs:
        #   @group_by: column name for grouping
        #   @cpuh_jobs: optional input of cpuh_jobs
        #   @bin_size: bins size of timeseries.
        #   @fg: (optional) figure for plots. Should have 4 subplots. if not provided a default will be created.
        #   @ax1,2,3,4: Optional. if not provided will be created and/or drawn from fg.
        '''
        #
        bin_size = bin_size or .1
        if fg is None:
            fg = plt.figure(figsize=(20,16))
            for k in range(1,5):
                fg.add_subplot(2,2,k)
            #
        #
        # NOTE: this might break if (fg is None), even if ax1,2,3,4 are provided)
        #ax1, ax2, ax3, ax4 = fg.get_axes()
        ax1 = ax1 or fg.get_axes()[0]
        ax2 = ax2 or fg.get_axes()[1]
        ax3 = ax3 or fg.get_axes()[2]
        ax4 = ax4 or fg.get_axes()[3]
        #
        ax1.grid()
        ax2.grid()
        #
        ax1.set_title('CPU Hours (per day)', size=16)
        ax2.set_title('Jobs (per day?)', size=16)
        ax3.set_title('CPU Hours', size=16)
        ax4.set_title('Jobs', size=16)
        #
        if cpuh_jobs is None:
            cpuh_jobs = self.get_cpu_hours_layer_cake(bin_size=bin_size, jobs_summary=jobs_summary, layer_field=group_by)
        #
        # shorthand pointers:
        cpuh = cpuh_jobs['cpu_hours']
        jobs = cpuh_jobs['jobs']
        #T = cpuh['time']
        #
        z_cpuh = plot_layer_cake(data=cpuh, layers=cpuh.dtype.names[1:], time_col='time', ax=ax1)
        z_jobs = plot_layer_cake(data=jobs, layers=cpuh.dtype.names[1:], time_col='time', ax=ax2)
        pie_cpuh_data = plot_pie(sum_data=self['Elapsed']*self['NCPUS'], slice_data=self[group_by], ax=ax3, wedgeprops=wedgeprops, autopct=autopct, slice_names=cpuh.dtype.names[1:])
        pie_jobs_data = plot_pie(sum_data=self['Elapsed'], slice_data=self[group_by], ax=ax4, wedgeprops=wedgeprops, autopct=autopct, slice_names=cpuh.dtype.names[1:])
        #
        ax1.legend(loc=0)
        ax2.legend(loc=0)
        #
        # TODO: return all the data sets in a dict?
        # return {'fg':fg, 'pie_cpuh_data':pie_cpuh_data, 'pie_jobs_data':pie_jobs_data, etc...}
        return fg

    def active_cpu_jobs_per_day_hour_report(self, qs=[.45, .5, .55], figsize=(14,10),
     cpu_usage=None, verbose=0, foutname=None, periodic_projection='rectilinear', polar=False):
        '''
        # 2x3 figure of instantaneous usage (active cpus, jobs per day, week,
        #  periodic/seasonal usage -- day of week, hour of day aggregates)
        #
        '''
        if cpu_usage is None:
            cpu_usage = self.cpu_usage
        #
        if polar:
            periodic_projection='polar'

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
    # 3 Oct 2022 yoder: skip Jobname; we wrestled a few hours with this because somebody decided
    #   it was a good idea to put a "|" in their jobname. Not much value in Jobname, it's user defined (and so
    #   unreliable), and expensive to store.
    # , 'Jobname'
    format_list_default = ['User', 'Group', 'GID', 'Account', 'JobID', 'JobIDRaw', 'partition', 'state', 'time', 'ncpus', 'ReqMem',
               'nnodes', 'Submit', 'Eligible', 'start', 'end', 'elapsed', 'SystemCPU', 'UserCPU',
               'TotalCPU', 'NTasks', 'CPUTimeRaw', 'Suspended', 'ReqTRES', 'AllocTRES', 'MaxRSS', 'AveRSS', 'AveVMsize', 'MaxVMsize',
               'MaxDiskWrite', 'MaxDiskRead', 'AveDiskWrite', 'AveDiskRead']
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
        self.attributes.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
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
        n_cpu = int(n_cpu or 4)
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
                # Handle jobs sumamry string/byte encoding. NOTE: this could probably be generally applied to all arrays.
                if ky == 'jobs_summary':
                    dtp = [(nm, d[0] if (isinstance(d,tuple) and d[0].startswith('|S')) else d ) for nm,d in ary.dtype.descr ]
                    self.__dict__[ky] = numpy.empty( (len(ary), ), dtype=dtp)
                    #
                    # DEBUG:
                    #print('** * ** ', self.__dict__[ky].dtype)
                    #print('** * ** ', numpy.shape(self.__dict__[ky]), len(ary), numpy.shape(self.__dict__[ky]))
                    #print('** * ** ', self.__dict__[ky].dtype)
                    for cl,tp in ary.dtype.descr:
                        #self.__dict__[ky][cl] = (ary[cl].astype(str)[:] if (isinstance(tp,tuple) and tp[0].startswith('|S') ) else ary[cl][:])
                        self.__dict__[ky][cl] = ([s.decode() for s in ary[cl]] if (isinstance(tp,tuple) and tp[0].startswith('|S') ) else ary[cl][:])
                #
                self.__dict__[ky]=ary[:]
            #
            # and attributes...:
            for ky,vl in fin.attrs.items():
                #TODO: do this elegandly (eg, with a [(ky,f), ...] pair list. For now...
                if ky in ('start_date', 'end_date') and isinstance(vl, str):
                    vl = mpd.num2date(mpd.datestr2num(vl))
                self.__dict__[ky] = vl
            #
            del ky,ary
        self.dt_mpd_epoch = self.compute_mpd_epoch_dt()
        #
        # Now, modify jobs_summary to use plain string, not byte-array, types?:
#        new_dtype = [(nm, d[0] if (isinstance(d,tuple) and d[0].startswith('|S')) else d )for s,d in self.jobs_summary.dtype.descr )]
#        new_js = numpy.array(len(self.jobs_summary), dtype=new_dtype)
#        for cl,d in new_js.dtype.descr:
#            if d.startswith('|S'):
#                new_js[cl] = self.jobs_summary[cl].
#        #
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
class SQUEUE_obj(object):
    # SQUEUE manager, principally to estimate realtime SLURM activity. Like SACCT_obj, but
    #  uses squeue. We *could* use SACCT_obj and just limit to --State=running,pending
    #. but it seems that sacct is much slower than squeue.
    #
    def __init__(self, partition='serc', format_fields_dict=None, squeue_prams=None, verbose=False):
        #
        # @squeue_prams: additional or replacement fields for squeue_fields variable, eg parameters
        #. to pass to squeue. Presently, --Format and --partition are specified. some options might also
        #. be allowed as regular inputs. Probably .update(squeue_prams) will be the last thing done, so
        #  will overried other inputs.
        #
        # TODO: add ***kwargs and handle syantax like SQU_{something} to add to squeue_fields, etc.
        #
        # TODO: use sinfo or SH_PART to get total resources (cpus, gpus) for partition=
        SP = SH_PART_obj()
        total_cpus = SP.get_total_cpus(partitions=partition)
        total_gpus = SP.get_total_gpus(partitions=partition)
        
        if format_fields_dict is None:
            format_fields_dict = default_SLURM_types_dict
#             format_fields.update({ky:int for ky in ['NODES', 'CPUS', 'TASKS', 'numnodes', 'numtasks', 'numcpus']})
#             #
#             ff_l = {ky.lower():val for ky,val in format_fields.items()}
#             ff_u = {ky.upper():val for ky,val in format_fields.items()}
#             format_fields.update(ff_l)
#             format_fields.update(ff_u)
#             del ff_l
#             del ff_u
        #
        squeue_fields = {'--Format': ['jobid', 'jobarrayid', 'partition', 'name', 'username', 'timeused',
                                      'timeleft', 'numnodes', 'numcpus', 'numtasks', 'state', 'nodelist'],
                      '--partition': [partition]
                      }
        if isinstance(squeue_prams, dict):
            squeue_fields.update(squeue_prams)
        #
        squeue_delim=';'
        sinfo_str = 'squeue '
        for ky,vl in squeue_fields.items():
            delim=' '
            if ky.startswith('--'):
                # long format
                delim='='
            #
            sinfo_str = '{} {}{}{}'.format(sinfo_str, ky, delim, f':{squeue_delim},'.join(vl))
        #
        if verbose:
            print('*** sinfo_str: {}'.format(sinfo_str))
            print('*** sinfo_ary: {}'.format(sinfo_str.split()))
        #
        # TODO:
        # port some of these bits to class-scope function calls, for class portability
        #
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})
        self.set_squeue_data()
    #
    def set_squeue_data(self):
        self.squeue_data = self.get_squeue_data()
        self.dtype       = self.squeue_data.dtype
    #
    def __getitem__(self, *args, **kwargs):
        return self.squeue_data.__getitem__(*args, **kwargs)
    def __setitem__(self, *args, kwargs):
        return self.squeue_data.__setitem__
    #
    def get_squeue_data(self, sinfo_str=None, squeue_delim=None, verbose=False):
        sinfo_str = sinfo_str or self.sinfo_str
        squeue_delim = squeue_delim or self.squeue_delim
        #
        print(f'** squeue: {sinfo_str}' )
        squeue_output = subprocess.run(sinfo_str.split(), stdout=subprocess.PIPE).stdout.decode().split('\n')
        #cols = squeue_output[0].split(squeue_delim)
        #
        # there is a smarter way to do this, eg:
        cols = squeue_output[0].split(squeue_delim)
        for k,cl in enumerate(cols):
            cl_0 = cl
            k_rep = 0
            while cols[k] in cols[0:k]:
                cols[k] = f'{cl}_{k_rep}'
        if verbose:
            print('** cols: ', cols)
        #
        return pandas.DataFrame(data=[[self.format_fields_dict.get(cl.lower(),str)(x)
                                  for x, cl in zip(rw.split(squeue_delim),
                                self.squeue_fields['--Format']) ]
                                 for rw in squeue_output[1:] if not len(rw.strip()) == 0],
                                   columns=cols).to_records()
    #
    def get_active_jobs(self, *args, **kwargs):
        # TODO: get some notes, comments, and function description in here. What is this for???
        # print('** DEBUG: args: {}'.format(args))
        if len(args)>=6:
            args[5]
        kwargs['do_jobs'] = True
        return self.get_active_cpus(*args, **kwargs)
    #
    def get_active_cpus(self, state='running,pending', do_refresh=False, state_data=None, ncpus=None, do_cpus=True, do_jobs=False):
        if do_refresh:
            self.set_squeue_data()
        #
        if isinstance(state,bytes):
            state=state.decode()
        if isinstance(state,str):
            state=state.split(',')
        #
        for k,s in enumerate(state):
            state[k] = s.upper()
        #
        if state_data is None:
            state_data = self['STATE']
        if ncpus is None:
            ncpus = self['CPUS']
        #
        ix = numpy.isin(state_data, state)
        n_jobs, n_cpus = numpy.sum(ix), numpy.sum(ncpus[ix])
        #
        if do_cpus and do_jobs:
            return (n_jobs, n_cpus)
        if do_cpus:
            return n_cpus
        if do_jobs:
            return n_jobs
    #
    def simple_wait_estimate(self, ncpus=1, max_cpus=None, do_refresh=False):
        # 
        max_cpus = max_cpus or self.total_cpus
        #
        active_cpus = self.get_active_cpus(state='running,pending', do_refresh=do_refresh)
        avail_cpus = max_cpus - active_cpus
        #
        if ncpus <= avail_cpus:
            return 0
        #
        cpus_needed = ncpus - avail_cpus
        #
        # now, spin down self['TIME_LEFT'] until we have enough CPUs to do our job. that TIME_LEFT is
        #. when our job should be available.
        #
        return None
    #
    def report_user_cpu_job_pies(self, state='RUNNING,PENDING', cpus_total=None, add_idle=True, ax1=None, ax2=None):
        #
        # TODO: use scontrol or sinfo to get an automagical cpus_total count.
        cpus_total = cpus_total or self.total_cpus
        #
        user_data = self.get_user_cpu_job_data(state=state, add_idle=add_idle)
        #
        if ax1 is None or ax2 is None:
            fg = plt.figure(figsize=(12,8))
            ax1 = fg.add_subplot(1,2,1)
            ax2 = fg.add_subplot(1,2,2)
        #
        ax1.pie(user_data['jobs'],  labels=user_data['user'], autopct='%.1f')
        ax2.pie(user_data['ncpus'], labels=user_data['user'], autopct='%.1f')
        #
        ax1.set_title('Jobs', size=16)
        ax2.set_title('CPUs', size=16)
        #
        #ax1.legend(loc=0)
        
        return user_data
    #
    def get_user_cpu_job_data(self, state='RUNNING,PENDING', cpus_total=None, add_idle=True):
        #partition = partition or self.partition
        cpus_total = cpus_total or self.total_cpus
        #
        if isinstance(state,bytes):
            state=state.decode()
        if isinstance(state,str):
            state = state.split(',')
        sq = self[numpy.isin(self['STATE'].astype(type(state)), state)]
        print('*** ', self['STATE'].astype(type(state))[0:10], '*** ', state)
        #
        users = list([u for u in set(sq['USER']) if not u is None])
        #if add_idle:
        #    users += ['idle']
        print('** users: ', users)
        #
        out_data = numpy.empty(shape=(len(users) + int(add_idle),),
                    dtype=[('user', f'U{max([len(s) for s in users])}'), ('jobs', '<f8'), ('ncpus', '<f8')])
        
        #out_data['jobs'] = numpy.zeros(len(users))
        #out_data['ncpus'] = numpy.zeros(len(users))
        #
        # NOTE: there is also a (faster) syntax to broadcast this, then sum on an axis.
        jobs  = [numpy.sum( (self['USER']==u) ) for u in users]
        ncpus = [numpy.sum( self['CPUS'][(self['USER']==u)] ) for u in users]
        #
        if add_idle:
            jobs += [0]
            ncpus += [cpus_total-numpy.sum(ncpus)]
            users += ['idle']
        #
        out_data['user']  = users
        out_data['jobs']  = jobs
        out_data['ncpus'] = ncpus
        #
        return out_data
#
class SPART_obj(object):
    # Python handler for SPART,or partition summary info kabob.
    # SPART looks like a third party tool to provide partition type info.
    # https://github.com/mercanca/spart
    #
    def __init__(self, options='gmft', format_fields_dict=None, verbose=False):
        '''
        # @options: spart options (see docs or man or whatever...). Can be list-like or string.
        # internally, this will get parsed into a list of unique letter values and passed like, '-g -m -f'
        #
        '''
        #
        options = [ (s.decode() if hasattr(s, 'decode') else s) for s in options]
        spart_str = 'spart {}'.format(' '.join(f'-{s}' for s in options) )
        self.SP = self.get_spart_data(spart_str=spart_str)
        #
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})

    def get_spart_data(self, spart_str=None, verbose=False):
        spart_str = spart_str or self.spart_str
        #
        squeue_output = subprocess.run(spart_str.split(), stdout=subprocess.PIPE).stdout.decode().split('\n')
        squeue_output = [rw.replace('|', '') for rw in squeue_output]
        #print('** DEBUG: ', squeue_output)
        #
        # the output is a bit of a mess, but we can put it together...
        for k,rw in enumerate(squeue_output):
            if rw.lstrip().startswith('QUEUE'):
                cols = rw.split()
                cols = [f'{c1}{c2}' for c1,c2 in zip(cols, squeue_output[k+1].split()) ]
                #
                break
            #
        data = [rw.split() for rw in squeue_output[k+2:] if not len(rw)==0]
        
        #return [cols] + data
        return pandas.DataFrame(data=data, columns=cols, index=[rw[0] for rw in data])
    #
    @property
    def cols(self):
        return self.SP.columns
    #
    def get_total_cpus(self, partitions='normal'):
        if isinstance(partitions, str):
            partitions = partitions.split(',')
        #
        return numpy.sum(self.SP['TOTALCORES'][partitions].to_numpy().astype(float))
    def get_total_gpus(self, partitions='gpu', verbose=False):
        # gpus in GRES, like: gpu:4(S:0)(2),gpu:8(S:0-3)(6)
        if isinstance(partitions, str):
            partitions = partitions.split(',')
        #
        gres = self.SP['GRES(NODE-COUNT)'][partitions].to_list()
        #
        # flatten and split by 'gpu:' entries:
        gres = [s for ss in gres for s in ss.split(',') if s.startswith('gpu')]
        #
        # TODO: The right way to do this is with regular expressions. but I hate regular expressions,
        #. so for now, we'll just hack it.
        #expr='gpu:\d+(S:.*)(\d+)'
        ngpus = 0
        for k,s in enumerate(gres):
            if verbose:
                print('** * DEBUG: gres: ', gres)
            n_gpu = s[4:s.index('(')]
            #print(f'** {s}, {n_gpu}')
            #
            if '(S:' in s:
                N_server = s[s.index('(',s.index('(')+1)+1:s.index(')', s.index(')')+1  ) ]
            else:
                try:
                    N_server = s[s.index('(')+1 : s.index(')')]
                except:
                    N_server = 1
            #
            #print(f'** [{k}] : {s} :: {n_gpu} :: {N_server} ')
            #
            ngpus += int(n_gpu)*int(N_server)
            
            
        
            
        return ngpus
#
class SH_PART_obj():
    # turns out that spart is not compatible with the new version(s) of SLURM.
    #. Kilian had to rewrite spart --> sh_part, and so shal we...
    #  So hopefully, we can limit this modification to __init__(), get_(sh/s)part_data(), and
    #. maybe some introductory variable definitions.
    #
    def __init__(self, format_fields_dict=None, verbose=False):
        sh_part_str = 'sh_part ' # I don't think there are any options...
        self.SP = self.get_data(sh_part_str=sh_part_str)
        #
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})
    #
    @property
    def cols(self):
        return self.SP.columns
    #
    def get_spart_data(self, *args, **kwargs):
        print('*** DEPRICATION: get_spart_data() removed from this class version.')
        return None
    #
    def get_total_cpus(self, partitions='normal'):
        if isinstance(partitions, str):
            partitions = partitions.split(',')
        return numpy.sum(self.SP['cpu_cores_total'][partitions].to_numpy())
    def get_total_gpus(self, partitions='normal'):
        if isinstance(partitions, str):
            partitions = partitions.split(',')
        return numpy.sum(self.SP['gpus_total'][partitions].to_numpy().astype(int))
    #
    def get_total(self, col='cpu_cores_total', partitions='normal'):
        if isinstance(partitions, str):
            partitions = partitions.split(',')
            # .astype(float)
        return numpy.sum(self.SP[col][partitions].to_numpy())
    #
    def get_col_to_dict(self, col='cpu_cores_total', partitions='normal'):
        if isinstance(partitions, str):
            partitions = partitions.split(',')
        return {p:v for (p,v) in zip(partitions, self.SP[col][partitions].to_numpy())}
    #
    def get_data(self, sh_part_str):
        '''
        # TODO: this is very much in progress...
        #
        # regrets... should have just called get_spart_data() get_data(), so we can override it.
        # we will correct that here and just nulify get_spart_data().
        #
        # sh_part has some nice human readable, hierarchical formatting, which we'll just get rid of
        #. to make it more machine readable.
        #. currently, row (of interest) sequence is:
        #. p_name, is_public, nodes_idle nodes_total, cpus_idle, cpus_total, cpus_queued, gpus_idle, gpus_total, gpus_queued, job_runtime_default, job_runtime_max, cores_per_node, mem_per_Node, gpus_per_node.
        #. so we'll be interested principally in cols [0,5,8] --> [name,cpus,gpus]
        '''
        #
        # build data type/funct_dict:
        col_f_dict = {'nodes_idle':int, 'nodes_total':int}
        col_f_dict.update({f'cpu_cores_{s}':int for s in ('idle', 'total', 'queued')})
        col_f_dict.update({f'gpus_{s}':int for s in ('idle', 'total', 'queued')})
        ##col_f_dict.update({f'job_runtime_{s}':str for s in ('default', 'maximum')})  # defer to default for now...
        #
        # don't think there are any options for sh_part, but one day there might be...
        sh_part_str = sh_part_str or self.sh_part_str
        sh_part_output = subprocess.run(sh_part_str.split(), stdout=subprocess.PIPE).stdout.decode().split('\n')
        #
        self.sh_part_output = sh_part_output
        #
        # Columns are built up wtih a littl hierarchy and human readability in mind:
        col_categories = [s.strip().lower() for s in sh_part_output[0].split('|') if not s.strip()=='']
        col_groups     = [[s.strip().lower() for s in x.split()] for x in sh_part_output[1].split('|') if not x.strip()=='']
        #
        columns = [f'{s1}_{s2}'.strip().replace(' ', '_') for s1,S in zip(col_categories, col_groups) for s2 in S]
        col_index = {k:cl for k,cl in enumerate(columns)}
        #self.columns = columns
        del col_categories
        del col_groups
        #
        # now, there will basically be up to 3 groups of partitions: public, not-public, owners. These are separated
        # by a dash-line ('--------...----')
        # start by just collecting the data; later we'll need to format it. Note also, if we want to get chici about it,
        # we'll need to split the per-node fields, eg: per-node_cores_min, per-node_cores_min.
        #
        #sh_part_data = [rw.lower().replace('|','').split() for rw in sh_part_output[2:] if not (rw.startswith('----') or len(rw.strip())==0)]
        sh_part_data = [[col_f_dict.get(col_index[k], str)(x) for k,x in enumerate(rw.lower().replace('|','').replace('*', '').split())] for rw in sh_part_output[2:] if not (rw.startswith('----') or len(rw.strip())==0)]
        #
        # If we want to use a numpy structured array, we have to do this (see below).
        #. unfortunately, this is one of those times that PANDAS works quite well, what with having
        #. built in column and row indices...
        # constructr dtype:
        # is there a less stupid way to do this??? I swear there is...
#         shp_dtype = []
#         for k, (x,cl) in enumerate(zip(sh_part_data[0], columns)):
#             if isinstance(x,bytes) or isinstance(x,str):
#                 # I think we can use 'S{n}' for both of these?
#                 shp_dtype += [tuple([cl, f'S{max([len(rw[k]) for rw in sh_part_data])}'])]
#             elif isinstance(x,int):
#                 shp_dtype += [tuple([cl, 'i8'])]
#             elif isinstance(x,float):
#                 shp_dtype += [tuple([cl, 'f8'])]
#             else:
#                 shp_dtype += [tuple([cl, 'S8'])]
#         #
#         return numpy.array([tuple(rw) for rw in sh_part_data], dtype=shp_dtype)
    
        return pandas.DataFrame(data=sh_part_data, columns=columns, index=[rw[0] for rw in sh_part_data])
#
# TODO: reports should be (may have already been?) moved to hpc_reports
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
        HPC_tex_obj = hpc_reports.Tex_Slides(Short_title=self.Short_title,
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
                #
                # just trap this so that it works. could throw a warning...
                # TODO: using xticks() should probably fix this; we can probalby get rid of the exception handling.
                try:
                    #lbls = [simple_date_string(mpd.num2date( float(s.get_text())) ) for s in ax.get_xticklabels()]
                    lbls = [simple_date_string(mpd.num2date(x + dt_epoch) ) for x in ax.get_xticks()]
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
    # both internal to this Class or external, and add slides to a report, or reports can be coded into the class itself.f
    #
    # General objective will be to borrow from, then Depricate and delete this and its hpc_lib. cousin, in favor of
    #  developing a more comprehensive and versatile version in hpc_reports.
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
        self.HPC_tex_obj = (TEX_obj or hpc_reports.Tex_Slides(Short_title=self.Short_title,
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
            #
            # TODO: FIXME: so... this grossly malfunctions for fs-scarp1. it picks up the second positional argument as text, instead of the first. aka:
            #
            # just trap this so that it works. could throw a warning...
            # TODO: may need to add epoch handling as well...
            try:
                lbls = [simple_date_string(mpd.num2date( float(x) ) ) for x in ax.get_xticks()]
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
#@numba.jit
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
#@numba.jit
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
    #   also de-dupe these rows (sort by job_id, then ???; then take max/min values for start, end, etc. times). But keep an eye on it; it could make mistakes.
    # TODO: add GPUs ? two ways: at the end: 1)do this task as presently defined, then compute GPUs from the whole set and add
    #   a column., 2) integrate a GPU column. into this workflow. To do it again... might just do this by-column anyway. there
    #   are performance (and other things too...) rleated strategies in this that -- to do it again, I might not repeat.
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
    # array of input_col, output_col, function to handle updates to jobs_summary array.
#   # js_col_f = [({output_col_name}, {input_col_name}, function), ... ]
    js_col_f = [('End',       'End', numpy.nanmax),\
                ('Start',     'Start', numpy.nanmax),\
                ('NCPUS',     'NCPUS', lambda x: numpy.nanmax(x).astype(int)),\
                ('NNodes',    'NNodes', lambda x: numpy.nanmax(x).astype(int)),\
                ('NTasks',    'NTasks', lambda x: numpy.nanmax(x).astype(int)),\
                ('MaxRSS',    'MaxRSS', numpy.nanmax),
                ('AveRSS',    'AveRSS', numpy.nanmax),
                ('MaxVMSize', 'MaxVMSize', numpy.nanmax),
                ('AveVMSize', 'AveVMSize', numpy.nanmax),
                ('MaxDiskWrite', 'MaxDiskWrite', numpy.nanmax),
                ('AveDiskWrite', 'AveDiskWrite', numpy.nanmax),
                ('MaxDiskRead',  'MaxDiskRead', numpy.nanmax),
                ('AveDiskRead',  'AveDiskRead', numpy.nanmax),
                ('NGPUs', 'AllocTRES', lambda x: numpy.nanmax(get_NGPUs(x)).astype(int) )
                ]
    #
    n_cpu = int(n_cpu or 1)
    js_dtype = numpy.dtype([(n, t[0] if isinstance(t,tuple) else t) for n,t in data.dtype.descr ] + [('NGPUs', '<i8')] )
    if verbose:
        print('*** computing jobs_summary func on {} cpu'.format(n_cpu))
    #
    if n_cpu > 1:
        # this should sort in place, which could cause problems downstream, so let's use an index:
        print('*** DEBUG: data[JobID]: {}, data[Submit]: {}'.format( (None in data['JobID']), (None in data['Submit']) ))
        #
        # Getting some "can't sort None type" errors:
        if verbose:
            print('*** DEBUG: NoneType evals...')
            print('*** DEBUG: type(data), len(data): ', type(data), len(data) )
            #print('*** DEBUG: nans in JobID?: ', numpy.isnan(data['JobID']).any() )
            print('*** DEBUG: nans in Submit?: ', numpy.isnan(data['Submit']).any() )
            print('*** DEGUG: Nones in JobID: ', numpy.sum([s is None for s in data['JobID']]))
        #
        # NOTE: sorts by named order=[] fields, then in order as they appear in dtype, to break any ties. When there are LOTS of records,
        #  and more pointedly, when there are dupes resulting from breaking up into sub-querries, all the real fields will tie,
        #  and eventually it will attempt to sort on a Nonetype which will break. The workaround is to add an index column, which is
        #  necessarily unique to a data set
        #
        #working_data = data[ix_s]
        working_data = data[:]
        working_data['index'] = numpy.arange(len(working_data))
        #
        if verbose:
            print('*** DEBUG: working_data.dtype: {}'.format(working_data.dtype))
            print('*** DEBUG: len(working_data[index]), len(unique(working_data[index])): {}, {}'.format(len(working_data['index']), len(numpy.unique(working_data['index']))))
        #
        # keep getting nonetype! seems like sorting on 'index' should pretty much sort it out, but no...
        #   this is possibly best called a "bug." It would seem that sort() sorts along the specified axes and then all the
        #   rest of them anyway (even after the sort is resolved). it is worth noting then that this might also result in a
        #   performance hit, so even though our solution -- to compute an index, looks stupid, it might actually be best practice.
        # seems sort of stupid, but this might work.
        #ix_temp = numpy.argsort(working_data[['JobID_parent', 'index']], order=['JobID_parent', 'index'])
        working_data = working_data[numpy.argsort(working_data[['JobID_parent', 'index']], order=['JobID_parent', 'index'])]
        #
        #ks = numpy.append(numpy.arange(0, len(data), step_size), [len(data)+1] )
        ks = numpy.array([*numpy.arange(0, len(data), step_size), len(data)+1])
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
        # NOTE: Movinng create(jobs_summary) to outside the if ncpus split...
        N = len(numpy.unique(data['JobID_parent']) )
        jobs_summary = numpy.zeros( (N, ), dtype=js_dtype)
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
    jobs_summary = numpy.zeros( shape=(len(job_ID_index), ), dtype=js_dtype)
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
        # another syntax challenge...
        #jobs_summary[k]=data[numpy.min(ks)]  # NOTE: these should be sorted, so we could use ks[0] if we need a speed boost.
        #
        # Again, an inflexible syntax challenge. the jobs_summary[] column index should be a list (?). In this case, we know the
        #  shape of a row from data[], so the first option should work. More generally, the assignment input needs to be a tuple.
        #this should work...
        jobs_summary[list(data.dtype.names)][k] = data[numpy.min(ks)]
        #jobs_summary[list(data.dtype.names)][k] = tuple(data[list(data.dtype.names)][numpy.min(ks)]
        jobs_summary['NGPUs'][k] = 0.
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
        # 5 july 2022, yoder: adding MaxRSS,AveRSS,AveVMsize,MaxVMsize,MaxDiskWrite,MaxDiskRead,AveDiskWrite,AveDiskRead
        #  to nanmax()
        # NOTE: The better way to assign these is to set up an array of [(col, func), ...], (see above):
#        jobs_summary[['End', 'Start', 'NCPUS', 'NNodes', 'NTasks','MaxRSS','AveRSS','MaxVMSize','AveVMSize','MaxDiskWrite','AveDiskWrite','MaxDiskRead','AveDiskRead','NGPUs']][k] = numpy.nanmax(sub_data['End']),\
#            numpy.nanmin(sub_data['Start']),\
#            numpy.nanmax(sub_data['NCPUS']).astype(int),\
#            numpy.nanmax(sub_data['NNodes']).astype(int),\
#            numpy.nanmax(sub_data['NTasks']).astype(int),\
#            numpy.nanmax(sub_data['MaxRSS']),\
#            numpy.nanmax(sub_data['AveRSS']),\
#            numpy.nanmax(sub_data['MaxVMSize']),\
#            numpy.nanmax(sub_data['AveVMSize']),\
#            numpy.nanmax(sub_data['MaxDiskWrite']),\
#            numpy.nanmax(sub_data['AveDiskWrite']),\
#            numpy.nanmax(sub_data['MaxDiskRead']),\
#            numpy.nanmax(sub_data['AveDiskRead'],\
#            numpy.nanmax(numpy.array([get_NGPUs(s) for s in sub_data['AllocTRES']]).astype(int))
#            )
        # this should do the trick for a compact format:
        cls = [n1 for n1, n2, f in js_col_f]
        jobs_summary[cls][k] = tuple([f(sub_data[n2]) for (n1, n2, f) in js_col_f])
        #
        del sub_data
    #
    if verbose:
        print('** DEBUG: jobs_summary.shape = {}'.format(jobs_summary.shape))
    #
    return jobs_summary
#
def get_NGPUs(alloc_tres=None):
        #
        # NOTE: This can optionally be depricated. GPU counts are not built into
        #  calc_jobs_summary(), so SACCT_obj['NGPUs'] will do the trick. At the same time,
        #  this is not hurting anything.
        #
        # TODO: get gpu types...
        return numpy.array([float(s.split('gpu=')[1].split(',')[0]) if 'gpu=' in s else 0. for s in numpy.array(alloc_tres).astype(str)])
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
    n_cpu = int(n_cpu or 1)
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
    #print(f'** DEBUG: t_min: {t_min}, t_max: {t_max}')
    #
    inputs.update({'t_min':t_min, 't_max':t_max})
    #
    # output container:
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
    #time_step_size = CPU_H['time'][1] - CPU_H['time'][0]
    #time_step_size = 1.0
    for k,t in enumerate(CPU_H['time']):
        # TODO: wrap this into a (@jit compiled) function to parallelize? or add a recursive block
        #  to parralize the whole function, using an index to break it up.
        # NOTE / TODO: we *could* add additional masks or indices here to, for example, filter out weekends and after-hours (focus on M-F,8-5)
        #   jobs... but it's still hard
        ix_k = numpy.where(numpy.logical_and(t_start<=t, t_end>=(t-bin_size) ))[0]
        #print('*** shape(ix_k): {}//{}'.format(ix_k.shape, numpy.sum(ix_k)) )
        #
        # If there are not jobs, ix_k will return empty; manually set it to 0s
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
        # 2022-08-08, yoder: added x/bin_size to CPU_HOURS calc, which I *think* fixed a sacaling problem.
        # TODO: this can be made faster (probably?) by basically counting all the middle elements and only doing max(), min(), and arithmetic
        #  on the leading and trailing elemnents. But if that breaks vectorization, we'll give up those gains.
        CPU_H[['cpu_hours', 'N_jobs']][k] = numpy.sum( (numpy.min([t*numpy.ones(N_ix), t_end[ix_k]], axis=0) -
            numpy.max( [(t-bin_size)*numpy.ones(N_ix), t_start[ix_k]], axis=0))*24.*(jobs_summary['NCPUS'])[ix_k] )/bin_size, N_ix
    #
    #CPU_H['cpu_hours']*=24.
    #print('*** returning CPU_H:: ', numpy.shape(CPU_H))
    return CPU_H
#
def active_jobs_cpu(n_points=5000, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, jobs_summary=None, verbose=None, mpp_chunksize=10000, nan_to=0., NCPUs=None):
    '''
    # TODO: Facilitate GPU computation...
    #  We could just add a GPUs column to this, but I think a more elegant solution is to add NCPUS
    #  as an optional parameter. Default behavior will be to get it from jobs_sumamry (current behavior);
    #  alternatively, an array like NGPUs can be provided.
    #
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
    # NOTE: see @mpp_chunksize below: speed performance was a big problem on Mazama, probably because it was IO limited? Not
    #   seeing the same problem(s) on Sherlock, so some of these more desperate discussions of performance optimization can probably
    #   be ignored.
    # @mpp_chunksize: batch size for MPP, in this case probably an apply_async(). Note that this has a significant affect on speed and parallelization
    #  performance, and optimal performance is probably related to maximal use of cache. For example, for moderate size data sets, the SPP speed might be ~20 sec,
    #  and the 2cpu, chunk_size=20000 might be 2 sec, then 1.5, 1.2, for 3,4 cores. toying with the chunk_size parameter suggests that cache size is the thing. It is
    #  probably worth circling back to see if map_async() can be made to work, since it probably has algorithms to estimate optimal chunk_size.
    # @nan_to={val} set nan values to {val}. If None, skip this step. Default is 0.
    # @NCPUs=None : vector of NCPU values. Default will be to use jobs_summary['NCPUS'], or we can provide -- for
    #  examle, a vector of NGPUs.
    '''
    #
    # TODO: modify this to allow a d_t designation, instead of n_points, so we can specify -- for example, an hourly sequence.
    # DEBUG: for now, just set n_cpu=1. we should be able to run this and fix the mpp later...
    #n_cpu=1
    #
    if NCPUs is None:
        NCPUs = jobs_summary['NCPUS']
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
    if len(t_start)==1 and verbose:
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
    if verbose:
        print(f'*** DEBUG: {n_points}, {bin_size}')
    output = numpy.zeros( n_points, dtype=[('time', '>f8'),
                ('N_jobs', '>f8'),
                ('N_cpu', '>f8')])
    #output['time'] = numpy.linspace(t_min, t_max - (t_max - t_min)/n_points, n_points)
    output['time'] = numpy.linspace(t_min, t_max, n_points)
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
        #
        # yoder, 2022-06-24: jobs_sumamry['NCPUS'] --> NCPUs
        #output['N_jobs'], output['N_cpu'] = process_ajc_rows(t=output['time'], t_start=t_start, t_end=t_end, NCPUs=jobs_summary['NCPUS'])
        output['N_jobs'], output['N_cpu'] = process_ajc_rows(t=output['time'], t_start=t_start, t_end=t_end, NCPUs=NCPUs)
        #
        if not nan_to is None:
            output['N_jobs'][numpy.isnan(output['N_jobs'])] = 0.
            output['N_cpu'][numpy.isnan(output['N_cpu'])]   = 0.
        #
        return output
        #
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
            # yoder, 2022-06-24: jobs_sumamry['NCPUS'] --> NCPUs
            #results = [P.apply_async(process_ajc_rows,  kwds={'t':output['time'], 't_start':t_start[k1:k2],
            #            't_end':t_end[k1:k2], 'NCPUs':jobs_summary['NCPUS'][k1:k2]} ) for k1, k2 in zip(ks_r[:-1], ks_r[1:])]
            results = [P.apply_async(process_ajc_rows,  kwds={'t':output['time'], 't_start':t_start[k1:k2],
                        't_end':t_end[k1:k2], 'NCPUs':NCPUs[k1:k2]} ) for k1, k2 in zip(ks_r[:-1], ks_r[1:])]
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
#

def compute_mpd_epoch_dt(test_date, yr_upper=3000, yr_lower=1000):
    '''
    # matplotlib.dates can use different epoch reference times. Older default was 0:000-12:32; newer is 0:1970-1-1 (I think).
    # use this to guess which epoch it is! Use this if you have data whose encoding is uncertain. It should give +/- a standard
    # value that will correct either encoding to the executing version of mpd .
    '''
    dt_epoch = 0.
    if isinstance(test_date, float) or isinstance(test_date, int):
        test_date = mpd.num2date(test_date)
    yr_test = test_date.year
    #
    #yr_test = mpd.num2date(test_date).year
    if yr_test > yr_upper:
        dt_epoch = -dt_mpd_epoch
    if yr_test < yr_lower:
        dt_epoch =  dt_mod_epoch
    #
    return dt_epoch
#
def flatten(A):
    # flagrantly stollen from:
    # https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt
#
def plot_layer_cake(data=None, layers=None, time_col='time', ax=None):
    '''
    # general handler for layer cake plots. Assume data is like [[time, {data cols}]].
    # Question: qualify this geometrically (dtype.names[1:]) or by value
    #  ([cl for cl in dtype.names if not cl=='time'] ?
    #  Or does that depend on the input type?
    #  we'll want to allow structured arrays and dict type.
    '''
    #
    if ax is None:
        fg = plt.figure(figsize=(10,8))
        ax = fg.add_subplot(1,1,1)
    fg = ax.get_figure()
    #
    if layers is None:
        if hasattr(data, 'dtype'):
            # assume dtype.names = (time, layer_1, layer_2, layer_3..., layer_n)
            # NOTE: the selection criteria for array,dict are not logically identical...
            layers = data.dtype.names[1:]
        else:
            layers = [ky for ky in data.keys() if not ky == time_col]
        #
    #
    T = data[time_col]
    #
    z = numpy.zeros(len(data))
    for lyr in layers:
        # there are more efficient ways to do this, but this will be fine...
        z_prev = z.copy()
        z += data[lyr]
        #
        ln, = ax.plot(T,z, ls='-', alpha=.8)
        clr = ln.get_color()
        ax.fill_between(T, z_prev, z, color=clr, alpha=.2, label=lyr)
    #
    fg.canvas.draw()
    dt_epoch = compute_mpd_epoch_dt(T[0])
    #
    # there may have been a reason for doing ticklabels this way, but I'm not seeing it. switching to
    #  more direct xticks(). Should find other instances as well..
#    lbls = [simple_date_string(mpd.num2date(float(fix_to_ascii(str(s.get_text()))) + dt_epoch) )
#              for s in ax.get_xticklabels()]
    lbls = [simple_date_string(mpd.num2date(x + dt_epoch) ) for x in ax.get_xticks()]
    ax.set_xticklabels(lbls)
    #
    return numpy.array([T,z]).transpose()
#
def get_pie_slices(sum_data, slice_data, slice_names=None):
    '''
    # basically, aggregate col_sum in groups of slice_names (which by default = unique(col_slices))
    # One tricky point is, to_encode() or not_to_encode().
    # For now, I think we can leave this to the user to control as input. By default, it will work
    # correctly, since the slice_names will be taken from data[col_slices]
    # @sum_data: a vector of data to sum by group
    # @slice_data: index is constructed from @slice data
    # @slice_names: slices. if None, all unique values from slice_data.
    #  eg, slice_k = sum(sum_data[slice_data==slice_name_k])
    '''
    #
    if slice_names is None:
        #slice_names = list(set(slice_data))
        slice_names = numpy.unique(slice_data)
    #
    # This is probably the 'right' way to do this:
    slice_names = numpy.array(slice_names).astype(numpy.array(slice_data).dtype)
    #
    #return numpy.array([[ky,numpy.sum(sum_data[slice_data==ky])] for ky in slice_names])
    return numpy.array( [(ky,numpy.sum(sum_data[slice_data==ky]) ) for ky in slice_names], dtype=[('name', f'{slice_names.dtype}'), ('value', '<f8')])
#
def plot_pie(sum_data, slice_data, slice_names=None, n_decimals=1, autopct='%1.1f%%', wedgeprops=None, ax=None):
    '''
    # make a pie chart using the get_pie_slices() helper function.
    # @sum_data: a vector of data to sum by group
    # @slice_data: index is constructed from @slice data
    # @slice_names: slices. if None, all unique values from slice_data.
    # @n_decimals: number of decimals in values for labels, eg slice_name: xxx.yy . If None, no values.
    #  eg, slice_k = sum(sum_data[slice_data==slice_name_k])
    #
    '''
    pi_data = get_pie_slices(sum_data=sum_data, slice_data=slice_data, slice_names=slice_names)
    #
    pi_lbls = pi_data['name'].astype(str)
    pi_vls  = pi_data['value'].astype(float)
    #
    if not n_decimals is None:
        #pi_lbls = numpy.array([f'{lbl}: {vl:.{n_decimals}f}' for lbl, vl in zip(pi_lbls, pi_vls)])
        pi_lbls = numpy.array([f'{lbl}: {vl:.{n_decimals}f}' for lbl, vl in zip(pi_lbls, pi_vls)])
    if ax is None:
        fg = plt.figure(figsize=(10,8))
        ax = fg.add_subplot(1,1,1)
    #
    pie_data = ax.pie(pi_vls, labels=pi_lbls, autopct=autopct, wedgeprops=wedgeprops)
    #
    return pie_data
#
def input_date_to_num(x):
    '''
    # multi-purpose input-date handler. Assume input is either a number (handle epoch problem?), properly formatted datestring,
    #  dtm.date, dtm.datetime, numpy.datetime64
    '''
    if isinstance(x, (dtm.date, dtm.datetime)):
        # NOTE: depending on reference epoch (for which the matplotlib.dates default has recently changed...),
        #  this may not be equivalent to dtm.date(time).toordinal()
        #x = dtm.datetime(x.year, x.month, x.day)
        return mpd.date2num(x)
    #
    if isinstance(x, (int, float)):
        # dt_mpd_epoch
        if mpd.num2date(x).year > 3000:
            return x - dt_mpd_epoch
        elif mpd.num2date(x).year < 1000:
            return x + dt_mpd_epoch
        else:
            return x
        #
    if isinstance(x, str):
        return mpd.datestr2num(x)
    #
    if isinstance(x, numpy.datetime64):
        return x.astype(float)
    #
#
def date_range_to_timeseries(t_start=0, t_end=None, x_val=None, t0=0., dt=.1):
    '''
    # convert a date range + X (two values) to a timeseries of X, eg:
    # {t_start=100.23, t_end=252.10, x_val=42}
    # @t_start, @t_end: start/end of input interval.
    # @x_val: data of interest. should be an int or float (one value)
    # @t0: start time of the sequence; phase shift of the sequence.
    # @dt: time step size.
    # Then, you can update a target sequence by using numpy.searchsorted() to find the insert/update point for
    #  the generated sequence. ie, k = numpy.searchsorted(t_master, xy[0][0]), then x_master[k-1:len(xy)] (or something
    #  very similar) should update the correct subsequence.
    # OPTIONS: return as XY sequence? return as {data:[], t0:f, k0:i} (ie, data, start time, and time seq. index)?
    #   returning a full XY sequence is more versatile and might mitigate mistakes downstream...
    '''
    #
    x_val = (x_val or 1.0)
    t_start = input_date_to_num(t_start)
    t_end   = input_date_to_num(t_end)
    #
    # get sequence end-points:
    # Note, these are the len(T) = len(X)+1 endpoints (includes the begining of the first bin).
    #print('** DEBUG: ', t_start, t_end, t0)
    ts_0   = dt*numpy.floor( (t_start - t0) /dt) + t0
    ts_end = dt*numpy.ceil( (t_end - t0) / dt) + t0
    #
    #print(f'** DEBUG: {N}')
    # Handle t_start == t_end condition. set min ts[-1] = ts[0]+dt.
    #ts = numpy.arange(ts_0-t0+dt, ts_end-t0+dt, dt)
    ts = numpy.arange(ts_0-t0+dt, max(ts_end-t0+dt, (ts_0-t0+2.*dt) ), dt)
    if len(ts) == 0:
        print('*** error: len=0. locals(): \n', locals() )
    xs = numpy.ones( len(ts) )*x_val
    xs[0]  *= (ts_0 + dt - t_start)/dt
    xs[-1] *= ( t_end - ts_end + dt)/dt
    #
    #return xs*X
    #
    # return numpy.array([ts, xs*X]).T
    return numpy.array([ts, xs]).T
    
def sinfo_ncpus_ngpus(partition='serc', as_tuple=False):
    '''
    # get number of cpu CPUs and gpu CPUs and GPUs from sinfo query. Return is like (?)
    #. {'cpu':N_cpu, 'gpu':N_gpu, 'gpu_cpu': N_gpu_cpu)}
    # what is the best output? {'gpu':{'gpu':n, 'cpu',m}} ?{'cpu', 'gpu', 'gpu_cpu'?}
    '''
    #
    partition=partition.replace(chr(32), '')
    #
    sinfo_str = f'sinfo -p {partition} -N --Format=cpus,gres'
    cpus_gres = squeue_output = subprocess.run(sinfo_str.split(),
                                               stdout=subprocess.PIPE).stdout.decode().split('\n')
    #
    cpus_gpus = {'cpu': 0, 'gpu': 0, 'gpu_cpu': 0}
    #
    for k,rw in enumerate(cpus_gres[1:]):
        #
        if len(rw)== 0:
            continue
        cpu,gre = rw.split()
        #
        n_gpu = None
        if 'gpu:' in gre:
            #print(f'** gre: {gre}')
            k0 = gre.index('gpu:')
            n_gpu = int(gre[k0+4:gre.index('(',k0) if '(' in gre[k0+4:] else len(gre)])
            #
            cpus_gpus['gpu'] += n_gpu
            cpus_gpus['gpu_cpu'] += int(cpu)
        else:
            cpus_gpus['cpu'] += int(cpu)
        #
    if as_tuple:
        return tuple(cpus_gpus[ky] for ky in ('cpu', 'gpu', 'gpu_cpu'))
    else:
        return cpus_gpus
    #
#
def fg_time_labels_to_dates(ax, dt_epoch=None):
    '''
    # convert an axis x-coordinates from float-date to date-date strings.
    '''
    fg = ax.figure
    x_ticks = ax.get_xticks()
    if dt_epoch is None:
        dt_epoch = compute_mpd_epoch_dt(x_ticks[0])
    lbls = [simple_date_string(mpd.num2date(x+dt_epoch)) for x in x_ticks]
    #
    fg.canvas.draw()
    ax.set_xticklabels(lbls)
    #
    return lbls
