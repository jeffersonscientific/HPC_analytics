# functions, classes, and other helper bits for HPC_analytics notebooks and other libs.
#
import os
import sys
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
#
class SACCT_data_handler(object):
    #
    # TODO: write GRES (gpu, etc. ) handler functions.
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
    def __init__(self, data_file_name, delim='|', max_rows=None, types_dict=None, chunk_size=1000, n_cpu=None,
                 n_points_usage=1000, verbose=0, keep_raw_data=False, h5out_file=None):
        '''
        # handler object for sacct data.
        #
        #@keep_raw_data: we compute a jobs_summary[] table, which probably needs to be an HDF5 object, as per memory
        #  requirements, But for starters, let's optinally dump the raw data. we shouldn't actually need it for
        #  anything we're doing right now. if it comes to it, maybe we dump it as an HDF5 or actually build a DB of some sort.
        '''
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
        if h5out_file is None:
            h5out_file='sacct_output.h5'
        self.h5out_file = h5out_file
        #
        self.headers = self.data.dtype.names
        # row-headers index:
        self.RH = {h:k for k,h in enumerate(self.headers)}
        #
        # local shorthand:
        data = self.data
        #
        jobs_summary = self.calc_jobs_summary()
        # BEGIN old jobs_summary code:
#        ix_user_jobs = numpy.array([not ('.batch' in s or '.extern' in s) for s in self.data['JobID']])
#        #
#        #job_ID_index = {ss:[] for ss in numpy.unique([s.split('.')[0]
#        #                                            for s in self.data['JobID'][ix_user_jobs] ])}
#        #
#        # this might be faster to get the job_ID_index since it only has to find the first '.':
#        job_ID_index = {ss:[] for ss in numpy.unique([s[0:(s+'.').index('.')]
#                                                    for s in self.data['JobID'][ix_user_jobs] ])}
#        #
#        #job_ID_index = dict(numpy.array(numpy.unique(self.data['JobID_parent'], return_counts=True)).T)
#        for k,s in enumerate(self.data['JobID_parent']):
#            job_ID_index[s] += [k]
#        #
#        #
#        if verbose:
#            print('Starting jobs_summary...')
#        # we should be able to MPP this as well...
#        jobs_summary = numpy.array(numpy.zeros(shape=(len(job_ID_index), )), dtype=data.dtype)
#        t_now = mpd.date2num(dtm.datetime.now())
#        for k, (j_id, ks) in enumerate(job_ID_index.items()):
#            jobs_summary[k]=data[numpy.min(ks)]  # NOTE: these should be sorted, so we could use ks[0]
#            #
#            # NOTE: for performance, because sub-sequences should be sorted (by index), we should be able to
#            #  use their index position, rather than min()/max()... but we'd need to be more careful about
#            #. that sorting step, and upstream dependencies are dangerous...
#            # NOTE: might be faster to index eact record, aka, skip sub_data and for each just,
#            #. {stuff} = max(data['End'][ks])
#            #
#            sub_data = data[sorted(ks)]
#            #
#            #jobs_summary[k]['End'] = numpy.max(sub_data['End'])
#            jobs_summary[k]['End'] = numpy.nanmax(sub_data['End'])
#            jobs_summary[k]['Start'] = numpy.nanmin(sub_data['Start'])
#            jobs_summary[k]['NCPUS'] = numpy.max(sub_data['NCPUS'])
#            jobs_summary[k]['NNodes'] = numpy.max(sub_data['NNodes'])
#        #
#        del sub_data
#        #
#        if verbose:
#            print('** DEBUG: jobs_summary.shape = {}'.format(jobs_summary.shape))
        #
        # END old jubs_summary code
        #
        if not keep_raw_data:
            del data
        #
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
        #
        # TODO: (re-)parallelize this...?? running continuously into problems with pickled objects being too big, so
        #   we need to be smarter about how we parallelize. Also, parallelization is just costing a lot of memory (like 15 GB/CPU -- which
        #   seems too much, so could be a mistake, but probalby not, since I'm pretty sure I ran a smilar job in SPP on <8GB).
        self.cpu_usage = self.active_jobs_cpu(n_cpu=n_cpu, mpp_chunksize=min(int(len(self.jobs_summary)/n_cpu), chunk_size) )
        self.weekly_hours = self.get_cpu_hours(bin_size=7, n_points=n_points_usage, n_cpu=n_cpu)
        self.daily_hours = self.get_cpu_hours(bin_size=1, n_points=n_points_usage, n_cpu=n_cpu)
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
    def calc_jobs_summary(self, data=None, verbose=None):
        '''
        # compute jobs summary from (raw)data
        '''
        if data is None: data = self.data
        if verbose is None: verbose=self.verbose
        #
        ix_user_jobs = numpy.array([not ('.batch' in s or '.extern' in s) for s in data['JobID']])
        #
        # NOTE: this can be a lot of string type data, so being efficient with string parsing makes a difference...
        job_ID_index = {ss:[] for ss in numpy.unique([s[0:(s+'.').index('.')]
                                                    for s in data['JobID'][ix_user_jobs] ])}
        #
        #job_ID_index = dict(numpy.array(numpy.unique(self.data['JobID_parent'], return_counts=True)).T)
        for k,s in enumerate(data['JobID_parent']):
            job_ID_index[s] += [k]
        #
        if verbose:
            print('Starting jobs_summary...')
        # we should be able to MPP this as well...
        jobs_summary = numpy.array(numpy.zeros(shape=(len(job_ID_index), )), dtype=data.dtype)
        t_now = mpd.date2num(dtm.datetime.now())
        for k, (j_id, ks) in enumerate(job_ID_index.items()):
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
            #jobs_summary[k]['End'] = numpy.max(sub_data['End'])
            jobs_summary[k]['End'] = numpy.nanmax(sub_data['End'])
            jobs_summary[k]['Start'] = numpy.nanmin(sub_data['Start'])
            jobs_summary[k]['NCPUS'] = numpy.max(sub_data['NCPUS'])
            jobs_summary[k]['NNodes'] = numpy.max(sub_data['NNodes'])
            #
            # aggregate on NTasks? The first row should be NTasks for the submitted job
            #jobs_summary[k]['NTasks'] = numpy.nanmax(sub_data['NTasks'])
            #
            # move this into the loop. weird things can happen with buffers...
            del sub_data
        #
        if verbose:
            print('** DEBUG: jobs_summary.shape = {}'.format(jobs_summary.shape))
        #
        return jobs_summary
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
        chunk_size = chunk_size or 100
        #
        n_cpu = n_cpu or self.n_cpu
        n_cpu = n_cpu or mpp.cpu_count()
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
            n_rws = 0
            # 
            # TODO: reevaluate readlines() vs for rw in...
            #
            # eventually, we might need to batch this.
            if n_cpu > 1:
                # TODO: use a context manager syntax instead of open(), close(), etc.
                P = mpp.Pool(n_cpu)
                #self.headers = headers
                #
                # TODO: how do we make this work with max_rows limit?
                # see also: https://stackoverflow.com/questions/16542261/python-multiprocessing-pool-with-map-async
                # for the use of P.map(), using a context manager and functools.partial()
                # TODO: use an out-of-class variation of process_row(), or modify the in-class so MPP does not need
                #   to pickle over the whole object, which breaks for large data sets. tentatively, use apply_async() and just pass
                #   headers, types_dict, and RH.
                #   def process_sacct_row(rw, delim='\t', headers=None, types_dict={}, RH={})
                results = P.map_async(self.process_row, fin, chunksize=chunk_size)
                P.close()
                P.join()
                data = results.get()
                #
                del results
                del P
            else:
                #
                data = [self.process_row(rw) for k,rw in enumerate(fin) if (max_rows is None or k<max_rows) ]
            #
            #all_the_data = fin.readlines(max_rows)
            #
#             data = []
#             #for j,rw in enumerate(fin):
#             for j,rw in enumerate(all_the_data[0:(len(all_the_data) or max_rows)]):
#                 #
#                 if rw[0:10] == headers_rw[0:10]:
#                     continue
#                 #
#                 #n_rws += 1
#                 ##if not (max_rows is None or max_rows<0) and j>max_rows:
#                 #if not (max_rows is None or max_rows<0) and n_rws>max_rows:
#                 #    break

#                 #print('*** DEBUG: ', rw)
#                 data += [rw.split(delim)[:-1]]
#                 #print('* * DEBUG: ', data[-1])
#                 #
#                 data[-1] = [None if vl=='' else types_dict.get(col,str)(vl)
#                     for k,(col,vl) in enumerate(zip(active_headers, data[-1]))]
#                 ##for k, (col,vl) in enumerate(zip(headers, data[-1])):
#                 #for k, (col,vl) in enumerate(zip(active_headers, data[-1])):
#                 #    #print('j,k [{}:{}]: {}, {}'.format(col,vl, j,k))
#                 #    #data[-1][k]=types_dict[col](vl)
#                 #    data[-1][k]=types_dict.get(col, str)(vl)
#                 #
                #
            #
#            del all_the_data, rw, k, cvol, vl
            #
            if verbose:
                print('** len: ', len(data))
                self.raw_data_len = len(data)
            #
            #self.data=data
            # TODO: asrecarray()?
            #self.data = pandas.DataFrame(data, columns=active_headers).to_records()
            #return data
            #
            # TODO: write a to_records() handler, so we don't need to use stupid PANDAS
            return pandas.DataFrame(data, columns=active_headers).to_records()
        #
    @numba.jit
    def process_row(self, rw):
        # use this with MPP processing:
        # ... but TODO: it looks like this is 1) inefficient and 2) breaks with large data inputs because I think it pickles the entire
        #  class object... so we need to move the MPP object out of class.
        #
        # use this for MPP processing:
        rws = rw.split(self.delim)
        #return [None if vl=='' else self.types_dict.get(col,str)(vl)
        #            for k,(col,vl) in enumerate(zip(self.headers, rw.split(self.delim)[:-1]))]
        return [None if vl=='' else self.types_dict.get(col,str)(vl)
                    for k,(col,vl) in enumerate(zip(self.headers, rws[:-1]))] + [rws[self.RH['JobID']].split('.')[0]]
    #
    # GIT Note: Deleting the depricated get_cpu_hours_2() function here. It can be recovered from commits earlier than
    #  31 Aug 2020.
    #
    #@numba.jit
    def get_cpu_hours_depricated(self, n_points=10000, bin_size=7., IX=None, t_min=None, jobs_summary=None, verbose=False):
        '''
        # DEPRICATION: this works, but is super memory intensive for large data sets, as per the use of vectorization and
        #  arrays, but possibly not actually benefiting from the massive memory consumption. Replacing with a simpler,
        #  loop-loop-like version.
        #
        # Get total CPU hours in bin-intervals. Note these can be running bins (which will cost us a bit computationally,
        #. but it should be manageable).
        # NOTE: By permitting jobs_summary to be passed as a param, we make it easier to do a recursive mpp operation
        #. (if n_cpu>1: {split jobs_summary into n_cpu pieces, pass back to the calling function with n_cpu=1
        '''
        #
        # use IX input to get a subset of the data, if desired. Note that this indroduces a mask (or something) 
        #. and at lest in some cases, array columns should be treated as 2D objects, so to access element k,
        #. ARY[col][k] --> (ARY[col][0])[k]
        if jobs_summary is None:
            # this actually is not working. having problems, i think, with multiple layers of indexing. we
            #. could try to trap this, or for now just force the calling side to handle it? I think we just have to make
            # a copy of the data...
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
        if verbose:
            print('** DEBUG: len(jobs_summary[ix]): {}'.format(len(jobs_summary)))
        #
        if len(jobs_summary)==0:
            return numpy.array([], dtype=[('time', '>f8'),
            ('t_start', '>f8'),
            ('cpu_hours', '>f8'),
            ('N_jobs', '>f8')])
        #
        # NOTE: See above discussion RE: [None] index and array column rank.
        t_start = jobs_summary['Start']
        #t_start = jobs_summary['Start'][0]
        #
        t_end = jobs_summary['End']
        #t_end = jobs_summary['End'][0]
        if verbose:
            print('** DEBUG: (get_cpu_hours) initial shapes:: ', t_end.shape, t_start.shape, jobs_summary['End'].shape )
        #
        t_end[numpy.logical_or(t_end is None, numpy.isnan(t_end))] = t_now
        #
        if verbose:
            print('** DEBUG: (get_cpu_hours)', t_end.shape, t_start.shape)
        #
        #
        if t_min is None:
            t_min = numpy.nanmin([t_start, t_end])
        t_max = numpy.nanmax([t_start, t_end])
        #
        # can also create X sequence like this, for minute resolution
        # X = numpy.arange(t_min, t_max, 1./(24*60))
        #
        X = numpy.linspace(t_min, t_max, n_points)
        #
        # in steps
        #IX_t = numpy.array([numpy.logical_and(t_start<=t, t_end>t) for t in X])
        if verbose:
            print('*** debug: starting IX_k')
        #
        # ... blocking these out in-line would save a lot of memory too...
        #IX_t = numpy.logical_and( X.reshape(-1,1)>=t_start, (X-bin_size).reshape(-1,1)<t_end )
        IX_k = [numpy.where(numpy.logical_and(x>=t_start, (x-bin_size)<t_end))[0] for x in X]
        #
        # we dont' actually do anything with this...
        #N_ix = numpy.array([len(rw) for rw in IX_k])
        #
        if verbose:
            print('*** DEBUG: IX_K:: ', IX_k[0:5])
            print('*** DEBUG: IX_k[0]: ', IX_k[0], type(IX_k[0]))
        #
        # This is the intermediate memory way to do this. We get away with it for raw data sets ~800 MB or so, but
        #  when we get up to the 3GB sets we find for Sherlock.owners, it gets out of control and we see memory usage
        #  peaking over 100GB... which is not ok. Turns out that simple loops can actually be faster with numba@jit compilation
        #  so let's see if we can do that...
        # ... but what happens if we just block this out one more level? are we taking a major memory hit from just the list comprehension?
        cpu_h = numpy.array([numpy.sum( (numpy.min([numpy.ones(len(jx))*x, t_end[jx]], axis=0) - 
                           numpy.max([numpy.ones(len(jx))*(x-bin_size), t_start[jx]], axis=0))*(jobs_summary['NCPUS'])[jx] )
                           for x,jx in zip(X, IX_k) ])
        #
        # convert to hours:
        cpu_h*=24.
        #
        if verbose:
            print('*** ', cpu_h.shape)
        #
        #
        return numpy.array([tuple(rw) for rw in numpy.array([X, X-bin_size, cpu_h, [len(rw) for rw in IX_k]]).T ], dtype=[('time', '>f8'), 
                                                                       ('t_start', '>f8'),
                                                                       ('cpu_hours', '>f8'),
                                                                       ('N_jobs', '>f8')])
    #
    #
    #@numba.jit
    def get_cpu_hours(self, n_points=10000, bin_size=7., IX=None, t_min=None, t_max=None, jobs_summary=None, verbose=False,
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
    #@numba.jit
    def active_jobs_cpu(self, n_points=5000, ix=None, bin_size=None, t_min=None, t_max=None, t_now=None, n_cpu=None, jobs_summary=None, verbose=None, mpp_chunksize=None):
        '''
        # @n_points: number of points in returned time series.
        # @bin_size: size of bins. This will override n_points
        # @t_min: start time (aka, bin phase).
        # @ix: an index, aka user=my_user
        '''
        # DEBUG: for now, just set n_cpu=1. we should be able to run this and fix the mpp later...
        #n_cpu=1
        #
        # try to trap for an empty set. ix can be booldan (len(ix)==len(data) ) or positional (len(ix) = len(subset) ).
        #. we can trap an all-false (empty) boolean index as sum(ix)==0 or a positonal index like len(x)==0,
        #. but there are corner cases to trying to trap both efficiently. so let's just trap positional cases for now.
        null_return = lambda: numpy.array([], dtype=[('time', '>f8'),('N_jobs', '>f8'),('N_cpu', '>f8')])
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
        output = numpy.zeros( n_points, dtype=[('time', '>f8'),
                    ('N_jobs', '>f8'),
                    ('N_cpu', '>f8')])
        output['time'] = numpy.linspace(t_min, t_max, n_points)
        #
#        if bin_size is None:
#            #X = numpy.linspace(t_min, t_max, n_points)
#            output = numpy.zeros( n_points, dtype=[('time', '>f8'),
#                                                    ('N_jobs', '>f8'),
#                                                    ('N_cpu', '>f8')])
#            output['time'] = numpy.linspace(t_min, t_max, n_points)
#        else:
#            X = numpy.arange(t_min, t_max, bin_size)
#            output = numpy.zeros( len(X), dtype=[('time', '>f8'),
#                                                ('N_jobs', '>f8'),
#                                                ('N_cpu', '>f8')])
#            output['time'] = X
#            del X
#        #
        #Ns = numpy.zeros(len(X))
        #Ns_cpu = numpy.zeros(len(X))
        #
        # This is a slick way to make an index, but It uses too much memory (unless
        #.  we want to have custom runs for high-mem HPC nodes... or have a go on one of the tool servers.
        # for now, I think this index is just too much...
        #IX_t = numpy.logical_and( X.reshape(-1,1)>=t_start, (X-bin_size).reshape(-1,1)<t_end )
        # the boolean index gets really big, so it makes sense to also (or alternatively) use an positional index:
        #
        # This also uses too much memory. we can use numpy.where() to convert an N length boolean index to an
        #  n length positional index, but it sill uses too much memory for large data sets:
        # use numpy.where() to convert a boolean index to a positional index
        #
        #IX_k = [numpy.where(numpy.logical_and(x>=t_start, x<=t_end) )[0] for x in X]
        #Ns = numpy.array([len(rw) for rw in IX_k])
        #
        # See above; there is a way to do this in full-numpy mode, but it will use most of the memory in the world,
        #.  so probably not reliable even on the HPC... and in fact likely not faster, since the matrices/vectors are
        #. ver sparse.
        #Ns_cpu = numpy.sum(jobs_summary['NCPUS'][IX_t], axis=1)
        #
        #Ns_cpu = numpy.array([numpy.sum(jobs_summary['NCPUS'][kx]) for kx in IX_k])
        # there should be a way to broadcast the array to indices, but I'm not finding it just yet...
        #Ns_cpu = numpy.sum(numpy.broadcast_to(jobs_summary['NCPUS'][0], (len(IX_k),len(jobs_summary) )
        #
        # Here's the blocked out loop-loop version. Vectorized was originally much faster, but maybe if we pre-set the array
        #
        if n_cpu==1:
            for j, t in enumerate(output['time']):
                ix_t = numpy.where(numpy.logical_and(t_start<=t, t_end>t))[0]
                #
                output[['N_jobs', 'N_cpu']][j] = len(ix_t), numpy.sum(jobs_summary['NCPUS'][ix_t])
                if verbose and j%1000==0:
                    print('*** PROGRESS: {}/{}'.format(j,len(output)))
                    print('*** ', output[j-10:j])
            #return output
        else:   
            with mpp.Pool(n_cpu) as P:
                # (star)map() method:
                
                #
                #print('*** executing in MPP mode...')
                # NOTE: large data sets break this if t_start, t_end inputs become very very large, which exceeds
                #. the pickling capacity, so for MPP we need to break up t_start, t_end. map_async() probably knows
                #  how to do this properly, or we can use apply_async and write a simple algorithm to limit the number
                #  of t_start, t_end records pased. we might even just do a loop-lop with @jit compilation... but first,
                #  let's try a map...
                # need to break this into small enough chunks to not break the pickle() indexing.
                n_procs = min(numpy.ceil(len(t_start)/n_cpu).astype(int), numpy.ceil(len(t_start)/mpp_chunksize).astype(int) )
                ks_r = numpy.linspace(0, len(t_start), n_procs+1).astype(int)
                #print('*** DEBUG: KS_R: ', ks_r)
                #
                #ks_r = numpy.linspace(t_min, t_max, n_cpu).astype(int)
                results = [P.apply_async(self.process_ajc_row_2,  kwds={'t':output['time'], 't_start':t_start[k1:k2],
                            't_end':t_end[k1:k2], 'NCPUs':jobs_summary['NCPUS'][k1:k2]} ) for k1, k2 in zip(ks_r[:-1], ks_r[1:])]
                #
                R = [r.get() for r in results]
                # join the pool? this throws a "pool still running" error... ???
                #P.join()
                #
            #print('** Shape(R): ', numpy.shape(results))
            # TODO: how do we numpy.sum(R) on the proper axis?
            for k_r, (r_nj, r_ncpu) in enumerate(R):
                #print('**** [{}]: {} ** {}'.format(k_r, r_nj[0:10], r_ncpu[0:10]))
                output['N_jobs'] += numpy.array(r_nj)
                output['N_cpu']  += numpy.array(r_ncpu)
            #
            del P, R, results
            #
        return output

    #
    def process_ajc_row_2(self, t, t_start, t_end, NCPUs):
        
        #print('*** DEBUG: {} ** {}'.format(len(t_start), len(t_end)))
        ix_t = numpy.logical_and(t_start<=t.reshape(-1,1), t_end>t.reshape(-1,1))
        #
        #N=numpy.sum(ix_t, axis=1)
        #C=numpy.sum(NCPUs.reshape(1,-1), axis=1)
        #
        # TODO: there is a proper syntax for the NCPUs calc...
        #r_val = numpy.array([numpy.sum(ix_t, axis=1), numpy.array([numpy.sum(NCPUs[j]) for j in ix_t])])
        #print('*** return from[{}], sh={} ** {}'.format(os.getpid(), r_val.shape, r_val[0][0:10]))
        return numpy.array([numpy.sum(ix_t, axis=1), numpy.array([numpy.sum(NCPUs[j]) for j in ix_t])])
    @numba.jit
    #def process_ajc_row(self,j, t_start, t_end, t, jobs_summary):
    def process_ajc_row(self, t):
        #print('*** DEBUG: t:: {}'.format(t) )
        ix_t = numpy.where(numpy.logical_and(self.t_start<=t, self.t_end>t))[0]
        #output[['N_jobs', 'N_cpu']][j] = len(ix_t), numpy.sum(jobs_summary['NCPUS'][ix_t])
        return len(ix_t), numpy.sum(self.ajc_js['NCPUS'][ix_t])
        #
    def get_wait_stats(self):
        # TODO: revise this to use a structured array, not a recarray.
        wait_stats = numpy.core.records.fromarrays(numpy.zeros((len(self.jobs_summary), 6)).T,
                                                   dtype=[('ncpus', '>i8'), ('mean', '>f8'), 
                                                                        ('median', '>f8'),  ('stdev', '>f8'),
                                                                        ('min', '>f8'),  ('max', '>f8')])
        #
        delta_ts = self.jobs_summary['Start'] - self.jobs_summary['Submit']
        #
        for j,k in enumerate(sorted(numpy.unique(self.jobs_summary['NCPUS']) ) ):
            #
            x_prime = delta_ts[numpy.logical_and(self.jobs_summary['NCPUS']==k, delta_ts>=0.)]
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
        #
        return wait_stats
    #
    def submit_wait_distribution(self, n_cpu=1):
        n_cpu = numpu.atleast_1d(n_cpu)
        #
        X = self.jobs_summary['Start'] - self.jobs_summary['Submit']
        #
        return numpy.array([n_cpu, [X[n] for n in n_cpu]]).T
    #
    def get_wait_times_per_ncpu(self, n_cpu):
        # get wait-times for n_cpu cpus. to be used independently or as an MPP worker function.
        #
        ix = self.jobs_summary['NCPUS']==n
        return self.jobs_summary['Start'][ix] - self.jobs_summary['Submit']
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
        Y = (self.jobs_summary['Start']- self.jobs_summary['Submit'] )*u_time
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
        if periodic_projection == 'polar':
            daily_Period = 2.0*scipy.constants.pi/24.
            weekly_Period = 2.0*scipy.constants.pi/7.
        else:
            daily_Period = 1.0
            weekly_Period = 1.0
        #
        #
        fg = plt.figure(figsize=figsize)
        #
        ax1 = fg.add_subplot('231')
        ax2 = fg.add_subplot('232', projection=periodic_projection)
        ax3 = fg.add_subplot('233', projection=periodic_projection)
        ax4 = fg.add_subplot('234')
        ax5 = fg.add_subplot('235', projection=periodic_projection)
        ax6 = fg.add_subplot('236', projection=periodic_projection)
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        #
        #qs = [.45, .5, .55]
        qs_s = ['q_{}'.format(q) for q in qs]
        #print('*** qs_s: ', qs_s)

        cpu_hourly = time_bin_aggregates(XY=numpy.array([cpu_usage['time'], 
                                                         cpu_usage['N_cpu']]).T, qs=qs)
        jobs_hourly = time_bin_aggregates(XY=numpy.array([cpu_usage['time'],
                                                          cpu_usage['N_jobs']]).T, qs=qs)
        #
        cpu_weekly = time_bin_aggregates(XY=numpy.array([cpu_usage['time']/7.,
                                                         cpu_usage['N_cpu']]).T, bin_mod=7., qs=qs)
        jobs_weekly = time_bin_aggregates(XY=numpy.array([cpu_usage['time']/7.,
                                                          cpu_usage['N_jobs']]).T, bin_mod=7., qs=qs)
        #
        ix_pst = numpy.argsort( (jobs_hourly['x']-7)%24)
        #
        X_tmp_hr = numpy.linspace(jobs_hourly['x'][0], jobs_hourly['x'][-1]+1,200)*daily_Period
        X_tmp_wk = numpy.linspace(jobs_weekly['x'][0], jobs_weekly['x'][-1]+1,200)*weekly_Period
        #
        hh1 = ax1.hist(sorted(cpu_usage['N_jobs'])[0:int(1.0*len(cpu_usage))], bins=25, cumulative=False)
        #ax2.plot(jobs_hourly['x'], jobs_hourly['mean'], ls='-', marker='o', label='PST')
        ln, = ax2.plot(jobs_hourly['x']*daily_Period, jobs_hourly['mean'][ix_pst], ls='-', marker='o', label='UTC')
        clr = ln.get_color()
        #
        #ax2.plot(jobs_hourly['x']*daily_Period, numpy.ones(len(jobs_hourly['x']))*numpy.mean(jobs_hourly['mean'][ix_pst]), ls='--', marker='', zorder=1, alpha=.7, label='mean', color=clr)
        ax2.plot(X_tmp_hr, numpy.ones(len(X_tmp_hr))*numpy.mean(jobs_hourly['mean'][ix_pst]), ls='--', marker='', zorder=1, alpha=.7, label='mean', color=clr)
        #
        ax3.plot(jobs_weekly['x']*weekly_Period, jobs_weekly['q_0.5'], ls='-', marker=
        'o', color=clr)
        #ax3.plot(jobs_weekly['x']*weekly_Period, jobs_weekly['mean'], ls='--', marker='', color=clr, label='mean')
        ax3.fill_between(jobs_weekly['x'], jobs_weekly[qs_s[0]], jobs_weekly[qs_s[-1]],
                         alpha=.1, zorder=1, color=clr)
        ax3.plot(X_tmp_wk, numpy.ones(len(X_tmp_wk))*numpy.mean(jobs_weekly['q_0.5']), ls='-.', alpha=.7, zorder=1, label='$<q_{0.5}$')
        #
        #
        hh4 = ax4.hist(cpu_usage['N_cpu'], bins=25)
        #ax5.plot(cpu_hourly['x'], cpu_hourly['mean'], ls='-', marker='o', label='PST')
        ln, = ax5.plot( cpu_hourly['x']*daily_Period, cpu_hourly['mean'][ix_pst], ls='-', marker='o', label='UTC')
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
        weekdays={ky:vl for ky,vl in zip(range(7), ['sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])}
        for ax in (ax3, ax6):
            #
            #print('*** wkly', weekly_Period*numpy.array(ax.get_xticks()))
            #ax.set_xticklabels(['', *[weekdays[int(x/weekly_Period)] for x in ax.get_xticks()[1:]] ] )
            #ax.set_xticks(numpy.arange(0,8)*weekly_Period)
            
            ax.set_xticks(cpu_weekly['x']*weekly_Period)
            ax.set_xticklabels(['sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
            
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
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        #####
        #
        with open(foutname, 'w') as fout:
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
        self.pdf_latex_out = subprocess.run(['pdflatex',  fname], cwd=output_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        # TODO: wrap this up into a class or function in HPC_lib.
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
            ax1 = plt.subplot('211')
            ax1.grid()
            ax1a = ax1.twinx()
            ax2 = plt.subplot('212', sharex=ax1)
            ax2.grid()
            #
            ax1.plot(act_jobs['time'], act_jobs['N_jobs'], ls='-', lw=2., marker='', label='Jobs', alpha=.5 )
            ax1a.plot(act_jobs['time'], act_jobs['N_cpu'], ls='--', lw=2., marker='', color='m',
                      label='CPUs', alpha=.5)
            #ax1a.set_title('Group: {}'.format(ky))
            fg.suptitle('Groaup: {}'.format(ky), size=16)
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

    def __init__(self, Short_title='HPC Analytics', Full_title='HPC Analitics Breakdown for Mazama',
                 out_path='output/HPC_analytics', tex_filename='HPC_analytics.tex', n_points_wkly_hrs=500,
                 fig_width_tex='.8', qs=[.45, .5, .55], fig_size=(10,8), SACCT_obj=None, max_rws=None ):
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ('self', '__class__')})
        #print('*** DEBUG: __init__: {}'.format(self.out_path))
        #
        #if self.groups is None:
        #    self.groups={'All':list(set(SACCT_obj.jobs_summary['User']))}
        #
        self.HPC_tex_obj = Tex_Slides(Short_title=self.Short_title,
                 Full_title=self.Full_title,
        foutname=os.path.join(out_path, tex_filename))
        #
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
        ax1 = plt.subplot('211')
        ax1.grid()
        ax1a = ax1.twinx()
        ax2 = plt.subplot('212', sharex=ax1)
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

def get_resercher_groups_dict():
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
    

def get_group_members(group_id):
    '''
    # use a subprocess call to getent to get all members of linux group group_id
    '''
    #
    #
    #sp_command="getent group | grep {}".format(group_id)
    sp_command = 'getent group'
    #
    # NOTE: in order to run a complex linux command, set shell=True. Some will say, however, that this is not recommended for security -- among other, reasons.
    # https://stackoverflow.com/questions/13332268/how-to-use-subprocess-command-with-pipes
    # The stackoverflow recommendation is to basically use two subprocesses (in this case, the first for getent, and then pipe the stdout to the stdin of a second
    #  process. For something simple, I don't see the advantage of that over just doing the grep or other operation in Python, which is a better environment for that
    #  sort of work anway. The advantage of the piped command is it is only a single call to the OS.
    #
    #print('subprocessing with shell=True...')
    # ...but it still breaks. likely composite subprocesses are not permitted, for security  purposes? make ssens...
    sp = subprocess.run(shlex.split(sp_command), shell=False, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    
    
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
    #. to convert the numerical date to weeks, so we'd want (t-t_0)%7, or we could use this function, but
    #. pass t=t_days/7., bin_mod=7, and really we'd want to do a phase shif to get DoW correctly.
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
    #return X_out

#
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
    #
    # TODO: better string handling please...
    #chrs = [[8211, '--'],]
    chrs = {8211:'--', 2013:'_'}
    for k,c in chrs.items():
        #print('** chrs:', k,c,s)
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
        
        
