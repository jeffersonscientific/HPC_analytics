# functions, classes, and other helper bits for HPC_analytics notebooks and other libs.
#
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
mazama_groups_ids = ['tgp', 'sep', 'clab', 'beroza', 'lnt', 'ds', 'nsd', 'oxyvibtest', 'cardamom', 'crustal', 'stress', 'das', 'ess', 'astro', 'seaf', 'oneill', 'modules', 'wheel', 'ds2', 'shanna', 'issm', 'fs-uq-zechner', 'esitstaff', 'itstaff', 'sac-eess164', 'sac-lab', 'sac-lambin', 'sac-lobell', 'fs-bpsm', 'fs-scarp1', 'fs-erd', 'fs-sedtanks', 'fs-sedtanks-ro', 'fs-supria', 'web-rg-dekaslab', 'fs-cdfm', 'suprib', 'cees', 'suckale', 'schroeder', 'thomas', 'ere', 'smart_fields', 'temp', 'mayotte-collab']
#
serc_user_ids = ['biondo', 'beroza', 'sklemp', 'harrisgp', 'gorelick', 'edunham', 'sagraham', 'omramom', 'aditis2', 'oneillm', 'jcaers', 'mukerji', 'glucia', 'tchelepi', 'lou', 'segall', 'horne', 'leift']
user_exclusions = ['myoder96', 'dennis']
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
    def __init__(self, data_file_name, delim='|', max_rows=None, types_dict=None, chunk_size=1000, n_cpu=None):
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
        n_cpu = (n_cpu or mpp.cpu_count() )
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
        # we should be able to MPP this as well...
        #jobs_summary = numpy.recarray(shape=(len(job_ID_index), self.data.shape[1]), dtype=data.dtype)
        jobs_summary = numpy.recarray(shape=(len(job_ID_index), ), dtype=data.dtype)
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
        print('** DEBUG: jobs_summary.shape = {}'.format(jobs_summary.shape))
        # Compute a job summary table. What columns do we need to aggregate? NCPUS, all the times: Start,
        #. End, Submit, Eligible. Note that the End time of the parent job often terminates before (only
        #. by a few seconds????) the End time of the last step, so we should probably just compute the
        #. Elapsed time. Start -> min(Start[]), End -> max(End[]), NCPU -> (NCPU of parent job or 
        #. max(NCPU)). For performance, we'll do well to not have to do logical, string-like operations --
        #  aka, do algebraic type operations.
        #
        #jobs_summary = numpy.array()
        #
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
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
                print('*** headers: ', headers)
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
    def get_cpu_hours(self, n_points=10000, bin_size=7., IX=None, t_min=None):
        '''
        # Get total CPU hours in bin-intervals. Note these can be running bins (which will cost us a bit computationally,
        #. but it should be manageable).
        '''
        #
        t_now = mpd.date2num( dtm.datetime.now() )
        #
        # use IX input to get a subset of the data, if desired. Note that this indroduces a mask (or something) 
        #. and at lest in some cases, array columns should be treated as 2D objects, so to access element k,
        #. ARY[col][k] --> (ARY[col][0])[k]
        jobs_summary = self.jobs_summary[IX]
        #
        # been wrestling with datetime types, as usual, so eventually decided maybe to just
        #. use floats?
        #t_start = self.jobs_summary['Start']
        t_start = jobs_summary['Start'][0]
        #
        #t_end = self.jobs_summary['End']
        t_end = jobs_summary['End'][0]
        #
        t_end[numpy.logical_or(t_end is None, numpy.isnan(t_end))] = t_now
        #
        print('** DEBUG: ', t_end.shape, t_start.shape)
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
        print('*** debug: starting IX_k')
        #
        #IX_t = numpy.logical_and( X.reshape(-1,1)>=t_start, (X-bin_size).reshape(-1,1)<t_end )
        IX_k = [numpy.where(numpy.logical_and(x>=t_start, (x-bin_size)<t_end))[0] for x in X]
        N_ix = numpy.array([len(rw) for rw in IX_k])
        #print('*** DEBUG: IX_k.shape: ', IX_k.shape)
        print('*** DEBUG: IX_K:: ', IX_k[0:5])
        print('*** DEBUG: IX_k[0]: ', IX_k[0], type(IX_k[0]))
        #
        # find empty sets: sum(bool)=0
        #Ns_ix = numpy.sum(IX_t, axis=1).astype(int)
        #print('*** debug: ', Ns_ix.shape)
        #
        #cpu_h[Ns_ix==0] = 0.
        #cpu_h = numpy.sum( numpy.min([X.reshape(-1,1), t_end[IX_k]] ) - numpy.max([(X-bin_size).reshape(-1,1), t_start[IX_k]] ) , axis=1)
        cpu_h = numpy.array([numpy.sum( (numpy.min([numpy.ones(len(ix))*x, t_end[ix]], axis=0) - 
                           numpy.max([numpy.ones(len(ix))*(x-bin_size), t_start[ix]], axis=0))*(jobs_summary['NCPUS'][0])[ix] )
                           for x,ix in zip(X, IX_k)])
        # convert to hours:
        cpu_h*=24.
                                      
        #cpu_h = numpy.sum([numpy.min([numpy.broadcast_to(X.reshape(-1,1), (len(X), len(ix))),
        #                                                 numpy.broadcast_to(t_end[ix], (len(X), len(ix))) ], axis=0 ) -   
        #                   numpy.max( [ numpy.broadcast_to( (X-bin_size).reshape(-1,1), (len(X), len(ix))),
        #                               numpy.broadcast_to(t_start[ix], (len(X), len(ix))) ], axis=0 )
        #                  for ix in IX_k], axis=1)
        #
        
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
        return numpy.core.records.fromarrays([X, X-bin_size, cpu_h, [len(rw) for rw in IX_k]], dtype=[('time', '>f8'), 
                                                                       ('t_start', '>f8'),
                                                                       ('cpu_hours', '>f8'),
                                                                       ('N_jobs', '>f8')])
        #return numpy.core.records.fromarrays([X, cpu_h], dtype=[('time', '>f8'), 
        #                                                             ('cpu_hours', '>f8')])
        #return numpy.array([X,cpu_h]).T
    #
    #@numba.jit
    def active_jobs_cpu(self, n_points=5000, ix=None, bin_size=None, t_min=None):
        '''
        # @n_points: number of points in returned time series.
        # @bin_size: size of bins. This will override n_points
        # @t_min: start time (aka, bin phase).
        # @ix: an index, aka user=my_user
        '''
        #
        #
        t_now = mpd.date2num( dtm.datetime.now() )
        #
        jobs_summary = self.jobs_summary[ix]
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
        #t_end = mpd.datestr2num(numpy.datetime_as_string(self.jobs_summary['End']))
        #
        #t_end = self.jobs_summary['End'].astype(dtm.datetime)
        #t_end[t_end is None]=numpy.datetime64(t_now)
        #t_end = mpd.date2num(t_end)
        #
        print('** DEBUG: ', t_end.shape, t_start.shape)
        #
        #
        if t_min is None:
            t_min = numpy.nanmin([t_start, t_end])
        t_max = numpy.nanmax([t_start, t_end])
        #
        # can also create X sequence like this, for minute resolution
        # X = numpy.arange(t_min, t_max, 1./(24*60))
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
        #IX_k = numpy.array([numpy.where(ix) for ix in IX_t])
        IX_k = [numpy.where(numpy.logical_and(x>=t_start, x<t_end))[1] for x in X]
        #IX_k = numpy.array([numpy.arange(len(t_start))[numpy.logical_and([x]>=t_start, [x]<t_end)] for x in X])
        #IX_k = numpy.array([[k for k, (t1,t2) in enumerate(zip(t_start, t_end)) if x>=t1 and x<t2] for x in X])
        #        
        #Ns = numpy.sum(IX_t, axis=1)
        Ns = numpy.array([len(rw) for rw in IX_k])
        #
        #
        # See above; there is a way to do this in full-numpy mode, but it will use most of the memory in the world,
        #.  so probably not reliable even on the HPC... and in fact likely not faster, since the matrices/vectors are
        #. ver sparse.
        #Ns_cpu = numpy.sum(jobs_summary['NCPUS'][IX_t], axis=1)
        
        Ns_cpu = numpy.array([numpy.sum(jobs_summary['NCPUS'][0][kx]) for kx in IX_k])
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
    
class SACCT_data_from_inputs(SACCT_data_handler):
    # a SACCT_data handler that loads input file(s). Most of the time, we don't need the primary
    #. data; we only want the summary data. We'll allow for both, default to summary...
    #. assume these are in a format that will natively import as a recarray (we'll use recarray->text 
    #. methods to produce them).
    #
    def __init__(self, summary_filename=None, primary_data_filename=None):
        pass
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
    #
    stats_output=[]
    for x in numpy.unique(X_mod):
        ix = X_mod==x
        this_Y = Y[ix]
        stats_output += [numpy.append([x, numpy.mean(this_Y), numpy.std(this_Y)],
                                      numpy.quantile(this_Y, qs))]
    #
    # TODO: convert this to a structured array. it looks (mostly) just like a record array, but it's not...
    #.  and it's faster...
    return numpy.core.records.fromarrays(numpy.array(stats_output).T, dtype=[('x', '>f8'), ('mean', '>f8'),
                                                        ('stdev', '>f8')] + 
                                         [('q_{}'.format(q), '>f8') for q in qs])
#

