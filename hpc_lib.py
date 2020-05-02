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
#
import numba
import pandas
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
        n_cpu = n_cpu or mpp.cpu_count()
        #
        self.__dict__.update({key:val for key,val in locals().items() if not key in ['self', '__class__']})
        #
        # especially for dev, allow to pass the DF object itself...
        if isinstance(data_file_name, str):
            self.data = self.load_sacct_data()
        else:
            self.data = data_file_name
            #self.data = self.data_df.values.to_list()
        self.headers = self.data.dtype.names
        self.RH = {h:k for k,h in enumerate(self.headers)}
        #
        # local shorthand:
        data = self.data
        #
        # sorting indices:
        # (maybe rename ix_sorting_{description} )
        index_job_id = numpy.argsort(self.data['JobID'])
        index_start  = numpy.argsort(self.data['Start'])
        index_end   = numpy.argsort(self.data['End'])
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
    @numba.jit
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
                del results, P
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
#         print('*** NCPUS_shape: ',jobs_summary['NCPUS'].shape) 
#         print('*** len(IX_k): ', Ns[0:5])
#         print('*** IX_k: ', IX_k[0:5])
#         print('*** IX?: ', numpy.sum(numpy.logical_and(X[0]>=t_start, X[0]<t_end)))
#         print('* * * ', X[0])
#         print('*** IX_where][{}]:: *. {} .*'.format(X[0], list(numpy.where(numpy.logical_and( X.reshape(-1,1)[0] >=t_start, X.reshape(-1,1)[0]<t_end))[1] ) ) )
        #
        # See above; there is a way to do this in full-numpy mode, but it will use most of the memory in the world,
        #.  so probably not reliable even on the HPC... and in fact likely not faster, since the matrices/vectors are
        #. ver sparse.
        #Ns_cpu = numpy.sum(jobs_summary['NCPUS'][IX_t], axis=1)
        
        Ns_cpu = numpy.array([numpy.sum(jobs_summary['NCPUS'][0][kx]) for kx in IX_k])
        # there should be a way to broadcast the array to indices, but I'm not finding it just yet...
        #Ns_cpu = numpy.sum(numpy.broadcast_to(jobs_summary['NCPUS'][0], (len(IX_k),len(jobs_summary) )
        
        
        #
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
    def get_submit_wait_timeofday(self, time_units='hours', qs=[.5, .75, .95]):
        #
        # TODO: allow units and modulus choices.
        #tu={'hour':'hours', 'hours':'hours', 'hr':'hours', 'hrs':'hours', 'min':'minutes',
        #    'minute':'minutes', 'minutes':'minutes', 
        #    'sec':'seconds', 'secs':'seconds', 'second':'seconds', 'seconds':'seconds'}
        #    
        #time_units = tu.get(time_units,time_units)
        #time_units_dict={'days':1., 'hours':24., 'minutes':float(24*60), 'seconds':float(24*3600)}
        #
        #tf = time_units_dict[time_units]
        #
        X = (self.jobs_summary['Submit']*24.)%24
        Y = (self.jobs_summary['Start']- self.jobs_summary['Submit'] )*24
        #
        # TODO: we were sorting for a reason, but do we still need to?
        ix = numpy.argsort(X)
        X = X[ix].astype(int)    # NOTE: at some point, we might actually want the non-int X values...
        Y = Y[ix]
        X0 = numpy.unique(X).astype(int)
        #        
        quantiles = numpy.array([numpy.quantile(Y[X==j], qs) for j in X0])
        
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

class SACCT_data_from_inputs(SACCT_data_handler):
    # a SACCT_data handler that loads input file(s). Most of the time, we don't need the primary
    #. data; we only want the summary data. We'll allow for both, default to summary...
    #. assume these are in a format that will natively import as a recarray (we'll use recarray->text 
    #. methods to produce them).
    #
    def __init__(self, summary_filename=None, primary_data_filename=None):
        pass
