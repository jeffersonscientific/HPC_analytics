# functions, classes, and other helper bits for HPC_analytics notebooks and other libs.
#
import os
import sys
import numpy
import scipy
#import matplotlib
import matplotlib.dates as mpd
#import pylab as plt
import datetime as dtm
import pytz
import multiprocessing as mpp
import pickle
import json
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
    return delim.join([str(x) for x in [dtm.year, dtm.month, dtm.day]])
#
def elapsed_time_2_day(tm_in, verbose=0):
    #
    # TODO: really??? why not just post the PL value?
    if tm_in in ( 'Partition_Limit', 'UNLIMITED' ):
        return None
    #
    tm_ins = tm_in.split('-')
    #
    if len(tm_ins)==0:
        return 0.
    if len(tm_ins)==1:
        days=0
        tm=tm_ins[0]
    if len(tm_ins)>1:
        days = tm_ins[0]
        tm = tm_ins[1]
    #
    #
    if verbose:
        try:
            h,m,s = tm.split(':')
        except:
            print('*** AHHH!!! error! tm_in: {}, tm_ins: {}, tm: {}'.format(tm_in, tm_ins, tm) )
            raise Exception("broke on elapsed time.")
    else:
        h,m,s = tm.split(':')
        
    #
    #return (float(days)*24.*3600. + float(h)*3600. + float(m)*60. + float(s))/(3600.*24)
    return float(days) + float(h)/24. + float(m)/(60.*24.) + float(s)/(3600.*24.) 
                       
def elapsed_time_2_sec(tm_in, verbose=0):
    #
    # TODO: really??? why not just post the PL value?
    if tm_in in ( 'Partition_Limit', 'UNLIMITED' ):
    #if tm_in.lower() in ( 'partition_limit', 'unlimited' ):
        return None
    #
    # guessing for now...
    tm_ins = tm_in.split('-')
    #
    if len(tm_ins)==0:
        return 0.
    if len(tm_ins)==1:
        days=0
        tm=tm_ins[0]
    if len(tm_ins)>1:
        days = tm_ins[0]
        tm = tm_ins[1]
    #
    
    if verbose:
        try:
            h,m,s = tm.split(':')
        except:
            print('*** AHHH!!! error! ', tm, tm_in)
            raise Exception("broke on elapsed time.")
    else:
        h,m,s = tm.split(':')
        
    #
    return float(days)*24.*3600. + float(h)*3600. + float(m)*60. + float(s)
#
def running_mean(X, n=10):
    return (numpy.cumsum(X)[n:] - numpy.cumsum(X)[:-n])/n
#
#
class SACCT_data_handler(object):
    #
    dtm_handler_default = str2date_num
    default_types_dict={'User':str, 'JobID':str, 'JobName':str, 'Partition':str, 'State':str, 'JobID_parent':str,
            'Timelimit':elapsed_time_2_day,
                'Start':dtm_handler_default, 'End':dtm_handler_default, 'Submit':dtm_handler_default,
                        'Eligible':dtm_handler_default,
                    'Elapsed':elapsed_time_2_day, 'MaxRSS':str,
            'MaxVMSize':str, 'NNodes':int, 'NCPUS':int, 'MinCPU':str, 'SystemCPU':str, 'UserCPU':str,
            'TotalCPU':str}
    #
    time_units_labels={'hour':'hours', 'hours':'hours', 'hr':'hours', 'hrs':'hours', 'min':'minutes','minute':'minutes', 'minutes':'minutes', 'sec':'seconds', 'secs':'seconds', 'second':'seconds', 'seconds':'seconds'}
    #    
    time_units_vals={'days':1., 'hours':24., 'minutes':24.*60., 'seconds':24.*3600.}
    #
    #
    def __init__(self, data_file_name, delim='|', max_rows=None, types_dict=None, chunk_size=1000, n_cpu=None, verbose=1):
        #
        if types_dict is None:
            #
            # handle all of the dates the same way. datetimes or numbers? numbers are easier for most
            #. purposes. Also, they don't morph unpredictably when transfered from one container to another.
            #. Dates are always a pain...
            #dtm_handler = str2date
            #dtm_handler = str2date_num
            types_dict=self.default_types_dict
        #
        #n_cpu = (n_cpu or mpp.cpu_count() )
        n_cpu = (n_cpu or 8)
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
        #
        # allow multiple inputs:
        # we need to work out duplicates before we can do this. for now, if we have to concatenate files, 
        #. let's either do it off-line or otherwise semi-manually.
        #
#         if not (isinstance(data_file_name, list) or isinstance(data_file_name, tuple) ):
#             data_file_name = [data_file_name]
#         #
#         for k, dfn in enumerate(data_file_name):
#             if isinstance(dfn, str):
#                 dta = self.load_sacct_data()
#             #
#             if k == 0:
#                 self.data = dta.copy()
#             else:
#                 self.data = numpy.append(self.data, dta)
#             #
#         #
#         # this is inefficient (we should do it before we make the arrays, or pass the array of filenames
#         #. to load_sacct_data(), but for now...
#         # de-dupe using numpy.unique(). Note rows need to be cast as tuple()
#         self.data = numpy.core.records.fromarrays(zip(*numpy.unique([tuple(rw) for rw in self.data])), dtype=self.data.dtype)

        # especially for dev, allow to pass the recarray object itself...
        if isinstance(data_file_name, str):
            self.data = self.load_sacct_data()
        else:
            self.data = data_file_name
            #self.data = self.data_df.values.to_list()
        #
        self.headers = self.data.dtype.names
        self.RH = {h:k for k,h in enumerate(self.headers)}
        #
        # local shorthand:
        data = self.data
        #
        # sorting indices:
        # (maybe rename ix_sorting_{description} )
        index_job_id = numpy.argsort(self.data['JobID'])
        #index_start  = numpy.argsort(self.data['Start'])
        #index_end   = numpy.argsort(self.data['End'])
        #
        # group jobs; compute summary table:
        ix_user_jobs = numpy.array([not ('.batch' in s or '.extern' in s) for s in self.data['JobID']])
        #
        #job_ID_index = {ss:[] for ss in numpy.unique([s.split('.')[0] 
        #                                            for s in self.data['JobID'][ix_user_jobs] ])}
        #
        # this might be faster to get the job_ID_index since it only has to find the first '.':
        job_ID_index = {ss:[] for ss in numpy.unique([s[0:(s+'.').index('.')]  
                                                    for s in self.data['JobID'][ix_user_jobs] ])}
        #
        #for k,s in enumerate(self.data['JobID']):
        #    job_ID_index[s.split('.')[0]] += [k]
        #    #group_index[s.split('.')[0]] = numpy.append(group_index[s.split('.')[0]], [k])
        #
        #job_ID_index = dict(numpy.array(numpy.unique(self.data['JobID_parent'], return_counts=True)).T)
        for k,s in enumerate(self.data['JobID_parent']):
            job_ID_index[s] += [k]
        #
        #
        if verbose:
            print('Starting jobs_summary...')
        # we should be able to MPP this as well...
        #jobs_summary = numpy.recarray(shape=(len(job_ID_index), self.data.shape[1]), dtype=data.dtype)
        #jobs_summary = numpy.recarray(shape=(len(job_ID_index), ), dtype=data.dtype)
        jobs_summary = numpy.array(numpy.zeros(shape=(len(job_ID_index), )), dtype=data.dtype)
        t_now = mpd.date2num(dtm.datetime.now())
        for k, (j_id, ks) in enumerate(job_ID_index.items()):
            jobs_summary[k]=data[numpy.min(ks)]  # NOTE: these should be sorted, so we could use ks[0]
            
            #
            # NOTE: for performance, because sub-sequences should be sorted (by index), we should be able to
            #  use their index position, rather than min()/max()... but we'd need to be more careful about
            #. that sorting step, and upstream dependencies are dangerous...
            # NOTE: might be faster to index eacy record, aka, skip sub_data and for each just,
            #. {stuff} = max(data['End'][ks])
            #
            sub_data = data[sorted(ks)]
            
            #jobs_summary[k]['End'] = numpy.max(sub_data['End'])
            jobs_summary[k]['End'] = numpy.nanmax(sub_data['End'])
            jobs_summary[k]['Start'] = numpy.nanmin(sub_data['Start'])
            jobs_summary[k]['NCPUS'] = numpy.max(sub_data['NCPUS'])
            jobs_summary[k]['NNodes'] = numpy.max(sub_data['NNodes'])
        #
        if verbose:
            print('** DEBUG: jobs_summary.shape = {}'.format(jobs_summary.shape))
        # Compute a job summary table. What columns do we need to aggregate? NCPUS, all the times: Start,
        #. End, Submit, Eligible. Note that the End time of the parent job often terminates before (only
        #. by a few seconds????) the End time of the last step, so we should probably just compute the
        #. Elapsed time. Start -> min(Start[]), End -> max(End[]), NCPU -> (NCPU of parent job or 
        #. max(NCPU)). For performance, we'll do well to not have to do logical, string-like operations --
        #  aka, do algebraic type operations.
        #
        # let's compute this by default...
        #
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
        self.cpu_usage = self.active_jobs_cpu()
    #
    #@numba.jit
    def load_sacct_data(self, data_file_name=None, delim=None, verbose=1, max_rows=None, chunk_size=None, n_cpu=None):
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
        #
        with open(data_file_name, 'r') as fin:
            headers_rw = fin.readline()
            if verbose:
                print('*** headers_rw: ', headers_rw)
            #headers = headers_rw[:-1].split(delim)[:-1]
            headers = headers_rw[:-1].split(delim)[:-1] + ['JobID_parent']
            #
            # make a row-handler dictionary, until we have proper indices.
            RH = {h:k for k,h in enumerate(headers)}
            self.RH = RH
            #
            #self.RH=RH
            #self.headers=headers
            #
            #delim_lines=fin.readline()
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
            #n_cpu = mpp.cpu_count()
            # eventually, we might need to batch this.
            if n_cpu > 1:
                # TODO: use a context manager syntax instead of open(), close(), etc.
                P = mpp.Pool(n_cpu)
                self.headers = headers
                #
                # see also: https://stackoverflow.com/questions/16542261/python-multiprocessing-pool-with-map-async
                # for the use of P.map(), using a context manager and functools.partial()
                results = P.map_async(self.process_row, fin, chunksize=chunk_size)
                P.close()
                P.join()
                data = results.get()
                #
                del results
                del P
            else:
                data = [self.process_row(rw) for rw in fin]
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
        #
        # use this for MPP processing:
        rws = rw.split(self.delim)
        #return [None if vl=='' else self.types_dict.get(col,str)(vl)
        #            for k,(col,vl) in enumerate(zip(self.headers, rw.split(self.delim)[:-1]))]
        return [None if vl=='' else self.types_dict.get(col,str)(vl)
                    for k,(col,vl) in enumerate(zip(self.headers, rws[:-1]))] + [rws[self.RH['JobID']].split('.')[0]]
    #
    #@numba.jit
    def get_cpu_hours(self, n_points=10000, bin_size=7., IX=None, t_min=None, jobs_summary=None, verbose=False):
        '''
        # Get total CPU hours in bin-intervals. Note these can be running bins (which will cost us a bit computationally,
        #. but it should be manageable).
        # NOTE: By permitting jobs_summary to be passed as a param, we make it easier to do a recursive mpp operation
        #. (if n_cpu>1: {split jobs_summary into n_cpu pieces, pass back to the calling function with n_cpu=1
        '''
        #
        t_now = mpd.date2num( dtm.datetime.now() )
        #
        # use IX input to get a subset of the data, if desired. Note that this indroduces a mask (or something) 
        #. and at lest in some cases, array columns should be treated as 2D objects, so to access element k,
        #. ARY[col][k] --> (ARY[col][0])[k]
        if jobs_summary is None:
            # this actually is not working. having problems, i think, with multiple layers of indexing. we
            #. could try to trap this, or for now just force the calling side to handle it? I think we just have to make
            # a copy of the data...
            jobs_summary = self.jobs_summary
            #jobs_summary = self.jobs_summary
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
        #IX_t = numpy.logical_and( X.reshape(-1,1)>=t_start, (X-bin_size).reshape(-1,1)<t_end )
        IX_k = [numpy.where(numpy.logical_and(x>=t_start, (x-bin_size)<t_end))[0] for x in X]
        N_ix = numpy.array([len(rw) for rw in IX_k])
        #
        if verbose:
            print('*** DEBUG: IX_K:: ', IX_k[0:5])
            print('*** DEBUG: IX_k[0]: ', IX_k[0], type(IX_k[0]))
        #
        cpu_h = numpy.array([numpy.sum( (numpy.min([numpy.ones(len(jx))*x, t_end[jx]], axis=0) - 
                           numpy.max([numpy.ones(len(jx))*(x-bin_size), t_start[jx]], axis=0))*(jobs_summary['NCPUS'])[jx] )
                           for x,jx in zip(X, IX_k) ])
#         # convert to hours:
        cpu_h*=24.
        #
        if verbose:
            print('*** ', cpu_h.shape)
        #
#         for k,t in enumerate(X):
#             ix_t = numpy.logical_and(t_start<=t, t_end>(t-bin_size) )
#             #print('*** shape(ix_t): {}//{}'.format(ix_t.shape, numpy.sum(ix_t)) )
#             #
#             N_ix = numpy.sum(ix_t)
#             if N_ix==0:
#                 cpu_hours[k] = t, t-bin_size,0.,0.
#                 continue
#             #
#             #print('** ** ', ix_t)
#             cpu_hours[k] = t, t-bin_size, numpy.sum( numpy.min([t*numpy.ones(N_ix), t_end[ix_t]], axis=0) - 
#                                numpy.max([(t-bin_size)*numpy.ones(N_ix), t_start[ix_t]]))*24., N_ix
#         #
        #return cpu_h
        #print('*** DB: ', N_ix[0:10])
        #print('*** shapes: ', [numpy.shape(s) for s in [X,X-bin_size, cpu_h, N_ix]])
#         return numpy.core.records.fromarrays([X, X-bin_size, cpu_h, [len(rw) for rw in IX_k]], dtype=[('time', '>f8'), 
#                                                                        ('t_start', '>f8'),
#                                                                        ('cpu_hours', '>f8'),
#                                                                        ('N_jobs', '>f8')])
#         #
        return numpy.array([tuple(rw) for rw in numpy.array([X, X-bin_size, cpu_h, [len(rw) for rw in IX_k]]).T ], dtype=[('time', '>f8'), 
                                                                       ('t_start', '>f8'),
                                                                       ('cpu_hours', '>f8'),
                                                                       ('N_jobs', '>f8')])
    #
    #@numba.jit
    def active_jobs_cpu(self, n_points=5000, ix=None, bin_size=None, t_min=None, verbose=0):
        '''
        # @n_points: number of points in returned time series.
        # @bin_size: size of bins. This will override n_points
        # @t_min: start time (aka, bin phase).
        # @ix: an index, aka user=my_user
        '''
        #
        # try to trap for an empty set. ix can be booldan (len(ix)==len(data) ) or positional (len(ix) = len(subset) ).
        #. we can trap an all-false (empty) boolean index as sum(ix)==0 or a positonal index like len(x)==0,
        #. but there are corner cases to trying to trap both efficiently. so let's just trap positional cases for now.
        null_return = lambda: numpy.array([], dtype=[('time', '>f8'),('N_jobs', '>f8'),('N_cpu', '>f8')])
        if (not ix is None) and len(ix)==0:
            #return numpy.array([(0.),(0.),(0.)], dtype=[('time', '>f8'), 
            #return numpy.array([], dtype=[('time', '>f8'),('N_jobs', '>f8'),('N_cpu', '>f8')])
            return null_return()
        #
        #
        t_now = mpd.date2num( dtm.datetime.now() )
        #
        jobs_summary = self.jobs_summary[ix]
        #
        if len(jobs_summary)==0:
            return null_return()
        #
        # been wrestling with datetime types, as usual, so eventually decided maybe to just
        #. use floats?
        #t_start = self.jobs_summary['Start']
        t_start = jobs_summary['Start']
        #
        #t_end = self.jobs_summary['End']
        t_end = jobs_summary['End']
        
        #t_end[t_end is None] = numpy.datetime64(t_now)
        t_end[numpy.logical_or(t_end is None, numpy.isnan(t_end))] = t_now
        #
        #
        #print('** DEBUG: ', t_end.shape, t_start.shape)
        #
        if t_min is None:
            t_min = numpy.nanmin([t_start, t_end])
        #
        print('** DEBUG: shapes:: {}, {}'.format(numpy.shape(t_start), numpy.shape(t_end)))
        if len(t_start)==1:
            print('** ** **: t_start, t_end: ', t_start, t_end)
        #
        # catch an empty set:
        # OR or AND condition???
        if len(t_start)==0 or len(t_end)==0:
            print('Exception: no data:: {}, {}'.format(numpy.shape(t_start), numpy.shape(t_end)))
            return null_return()
        #
        t_max = numpy.nanmax([t_start, t_end])
        #
        if bin_size is None:
            X = numpy.linspace(t_min, t_max, n_points)
        else:
            X = numpy.arange(t_min, t_max, bin_size)
        #
        Ns = numpy.zeros(len(X))
        Ns_cpu = numpy.zeros(len(X))
        #
        # This is a slick way to make an index, but It uses too much memory (unless
        #.  we want to have custom runs for high-mem HPC nodes... or have a go on one of the tool servers.
        # for now, I think this index is just too much...
        #IX_t = numpy.logical_and( X.reshape(-1,1)>=t_start, (X-bin_size).reshape(-1,1)<t_end )
        # the boolean index gets really big, so it makes sense to also (or alternatively) use an positional index:
        #
        # really a pain to get this to handle an empty set...
        # use numpy.where() to convert a boolean index to a positional index
        #IX_k = numpy.array([numpy.where(ix) for ix in IX_t])
        # so... was getting the index in [1], but now getting an index error and test show the indes in [0]...
        IX_k = [numpy.where(numpy.logical_and(x>=t_start, x<=t_end) )[0] for x in X]
        #IX_k = numpy.array([numpy.arange(len(t_start))[numpy.logical_and([x]>=t_start, [x]<t_end)] for x in X])
        #IX_k = numpy.array([[k for k, (t1,t2) in enumerate(zip(t_start, t_end)) if x>=t1 and x<t2] for x in X])
        #
        # NOTE: be sure we know if we're using boolean or positional indices. Note we could make this more robust (tolerant, and correct) for both by doing
        #  something like, len(jobs_summary[{any_col}][rw] , but skipping the broadcast should give us a significant performance boost.
        #Ns = numpy.sum(IX_t, axis=1)
        Ns = numpy.array([len(rw) for rw in IX_k])
        #
        # See above; there is a way to do this in full-numpy mode, but it will use most of the memory in the world,
        #.  so probably not reliable even on the HPC... and in fact likely not faster, since the matrices/vectors are
        #. ver sparse.
        #Ns_cpu = numpy.sum(jobs_summary['NCPUS'][IX_t], axis=1)
        #
        Ns_cpu = numpy.array([numpy.sum(jobs_summary['NCPUS'][kx]) for kx in IX_k])
        # there should be a way to broadcast the array to indices, but I'm not finding it just yet...
        #Ns_cpu = numpy.sum(numpy.broadcast_to(jobs_summary['NCPUS'][0], (len(IX_k),len(jobs_summary) )
        #
        # Here's the blocked out loop-loop version. Vectorized is much faster.
#         for j, t in enumerate(X):
#             ix_t = numpy.logical_and(t_start<=t, t_end>t)
#             #
#             # TODO: weight jobs, ncpu by fraction of time active during bin interval.
#             #
#             Ns[j] = numpy.sum(ix_t.astype(int))
#             Ns_cpu[j] = numpy.sum(jobs_summary['NCPUS'][ix_t])
        # 
        return numpy.core.records.fromarrays([X, Ns, Ns_cpu], dtype=[('time', '>f8'), 
                                                                     ('N_jobs', '>f8'),
                                                                     ('N_cpu', '>f8')])
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
        #pth = pathlib.Path(outputy_path)
        #
        # in the end, this is quick and easy, but for a reason -- it does not work very well.
        #. same goes for the quick and easy PANDAS file read method. it casts strings and other
        #. unknowns into Object() classes, which we basically can't do anything with. to reliably
        #. dump/load data to/from file, we need to handle strings, etc. but maybe we just pickle()?
        return self.data.tofile(outputy_path,
                                        format=('%s{}'.format(delim)).join(self.headers), sep=delim)
    #
    def export_summary_data(self, output_path='data/SACCT_summaruy_data.csv', delim='\n'):
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
    def get_submit_wait_timeofday(self, time_units='hours', qs=[.5, .75, .95]):
        '''
        # wait times as a functino fo time-of-day. Compute various quantiles for these values.
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
        ix = numpy.argsort(X)
        X = X[ix].astype(int)    # NOTE: at some point, we might actually want the non-int X values...
        Y = Y[ix]
        X0 = numpy.unique(X).astype(int)
        #        
        quantiles = numpy.array([numpy.quantile(Y[X==j], qs) for j in X0])
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
     cpu_usage=None, verbose=0, foutname=None):
        '''
        # 2x3 figure of instantaneous usage (active cpus, jobs per day, week)
        '''
        if cpu_usage is None:
            cpu_usage = self.cpu_usage
        #
        fg = plt.figure(figsize=figsize)
        #
        ax1 = fg.add_subplot('231')
        ax2 = fg.add_subplot('232')
        ax3 = fg.add_subplot('233')
        ax4 = fg.add_subplot('234')
        ax5 = fg.add_subplot('235')
        ax6 = fg.add_subplot('236')
        axs = [ax1, ax2, ax3, ax4, ax5, ax6]
        [ax.grid() for ax in axs]
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
        hh1 = ax1.hist(sorted(cpu_usage['N_jobs'])[0:int(1.0*len(cpu_usage))], bins=25, cumulative=False)
        #ax2.plot(jobs_hourly['x'], jobs_hourly['mean'], ls='-', marker='o', label='PST')
        ax2.plot((jobs_hourly['x']), jobs_hourly['mean'][ix_pst], ls='-', marker='o', label='UTC')
        ax3.plot(jobs_weekly['x'], jobs_weekly['q_0.5'], ls='-', marker='o', color='b')
        ax3.plot(jobs_weekly['x'], jobs_weekly['mean'], ls='--', marker='', color='b')
        ax3.fill_between(jobs_weekly['x'], jobs_weekly[qs_s[0]], jobs_weekly[qs_s[-1]],
                         alpha=.1, zorder=1, color='b')
        #
        #
        hh4 = ax4.hist(cpu_usage['N_cpu'], bins=25)
        #ax5.plot(cpu_hourly['x'], cpu_hourly['mean'], ls='-', marker='o', label='PST')
        ax5.plot( (cpu_hourly['x']), cpu_hourly['mean'][ix_pst], ls='-', marker='o', label='UTC')
        ax6.plot(cpu_weekly['x'], cpu_weekly['q_0.5'], ls='-', marker='o', color='b')
        ax6.plot(cpu_weekly['x'], cpu_weekly['mean'], ls='--', marker='', color='b')
        #
        # TODO: can we simplyfy this qs syntax?
        ax6.fill_between(cpu_weekly['x'], cpu_weekly[qs_s[0]], cpu_weekly[qs_s[-1]], alpha=.1, zorder=1, color='b')
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
        for ax in (ax3, ax6):
            ax.set_xticklabels(['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        #
        if not foutname is None:
            pth,nm = os.path.split(foutname)
            if not os.path.isdir(pth):
                os.makedirs(pth)
            #
            plt.savefig(foutname)
        #
        return {'cpu_hourly':cpu_hourly, 'jobs_hourly':jobs_hourly, 'cpu_weekly':cpu_weekly, 'jobs_weekly':jobs_weekly}

    
class SACCT_data_from_inputs(SACCT_data_handler):
    # a SACCT_data handler that loads input file(s). Most of the time, we don't need the primary
    #. data; we only want the summary data. We'll allow for both, default to summary...
    #. assume these are in a format that will natively import as a recarray (we'll use recarray->text 
    #. methods to produce them).
    #
    def __init__(self, summary_filename=None, primary_data_filename=None):
        pass
#
class Tex_Slides(object):
    def __init__(self, template_json='tex_components/EARTH_Beamer_template.json',
                 Short_title='', Full_title='Title',
               author='Mark R. Yoder, Ph.D.',
               institution='Stanford University, School of Earth', institution_short='Stanford EARTH',
              email='mryoder@stanford.edu', foutname='output/HPC_analytics/HPC_analytics.tex',
              project_tex=None ):
        #
        # TODO: test saving/reloading presentation_tex. A lso, save inputs, so we can completely reload
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
                 add_all_groups=True,
                 fig_width='.8', qs=[.45, .5, .55], SACCT_obj=None, max_rws=None, group_exclusions=group_exclusions ):
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ('self', '__class__')})
        #print('*** DEBUG: __init__: {}'.format(self.out_path))
        #
        self.make_report()
        
    
    def make_report(self, out_path=None, tex_filename=None, qs=None, max_rws=None, fig_width=None, verbose=1):

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
        groups = self.groups
        if max_rws is None:
            max_rws = self.max_rws
        fig_width = str(fig_width or self.fig_width)
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
        #. it wil build and search an index, butwe will benefit from simplicity.
        #
        if self.add_all_groups:
            #and not "all" in [s.lower() for s in groups.keys()]:
            groups['All'] = list(set([s for rw in groups.values() for s in rw]))

        print('keys: ', groups.keys() )
        fig_size=tuple((12,9))
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
            gpr_tex = ky.replace('\\_', '\_')
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
            wkly_hrs = SACCT_obj.get_cpu_hours(bin_size=7, n_points=500, IX=ix, verbose=0)
            act_jobs = SACCT_obj.active_jobs_cpu(bin_size=None, t_min=None, ix=ix, verbose=0)
            #
            # DEBUG:
            #print('*** weekly: ', wkly_hrs)
            #print('*** act_jobs: ', act_jobs)
            if len(act_jobs)==0:
                print('Group: {}:: no records.'.format(ky))
                continue
            #
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
            ax1a.set_title('Group: {}'.format(ky))
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
                    print('** WARNING: failed writing date text labels:: {}'.format(lbls = "[simple_date_string(mpd.num2date(max(1, float(s.get_text())) ) ) for s in ax.get_xticklabels()]" ))
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
                                                figsize=fig_size, cpu_usage=act_jobs,foutname=None)
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
#######
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
def time_bin_aggregates(XY, bin_mod=24, qs=[.25, .5, .75]):
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
        stats_output += [numpy.append([x, numpy.mean(this_Y), numpy.std(this_Y)],
                                      numpy.quantile(this_Y, qs))]
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
    return X_out

#

