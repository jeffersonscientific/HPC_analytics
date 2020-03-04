# functions, classes, and other helper bits for HPC_analytics notebooks and other libs.
#
import numpy
import scipy
#import matplotlib
import matplotlib.dates as mpd
#import pylab as plt
import datetime as dtm
import multiprocessing as mpp
import pickle
#

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
def elapsed_time_2_day(tm_in):
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
    h,m,s = tm.split(':')
    #try:
    #    h,m,s = tm.split(':')
    #except:
    #    print('*** AHHH!!! error! ', tm, tm_in)
    #    raise Exception("broke on elapsed time.")
        
    #
    #return (float(days)*24.*3600. + float(h)*3600. + float(m)*60. + float(s))/(3600.*24)
    return float(days) + float(h)/24. + float(m)/(60.*24.) + float(s)/(3600.*24.) 
                       
def elapsed_time_2_sec(tm_in):
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
    h,m,s = tm.split(':')
    #try:
    #    h,m,s = tm.split(':')
    #except:
    #    print('*** AHHH!!! error! ', tm, tm_in)
    #    raise Exception("broke on elapsed time.")
        
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
    default_types_dict={'User':str, 'JobID':str, 'JobName':str, 'Partition':str, 'State':str,
            'Timelimit':elapsed_time_2_day,
                'Start':dtm_handler_default, 'End':dtm_handler_default, 'Submit':dtm_handler_default,
                        'Eligible':dtm_handler_default,
                    'Elapsed':elapsed_time_2_day, 'MaxRSS':str,
            'MaxVMSize':str, 'NNodes':int, 'NCPUS':int, 'MinCPU':str, 'SystemCPU':str, 'UserCPU':str,
            'TotalCPU':str}
    #
    def __init__(self, data_file_name, delim='|', max_rows=None, types_dict=None, chunk_size=1000):
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
        index_endf   = numpy.argsort(self.data['End'])
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
        for k,s in enumerate(self.data['JobID']):
            job_ID_index[s.split('.')[0]] += [k]
            #group_index[s.split('.')[0]] = numpy.append(group_index[s.split('.')[0]], [k])
        #
        # we should be able to MPP this as well...
        #jobs_summary = numpy.recarray(shape=(len(job_ID_index), self.data.shape[1]), dtype=data.dtype)
        jobs_summary = numpy.recarray(shape=(len(job_ID_index), ), dtype=data.dtype)
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
            jobs_summary[k]['End'] = numpy.max(sub_data['End'])
            jobs_summary[k]['Start'] = numpy.min(sub_data['Start'])
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
    def load_sacct_data(self, data_file_name=None, delim=None, verbose=0, max_rows=None, chunk_size=None):
        if data_file_name is None:
            data_file_name = self.data_file_name
        if delim is None:
            delim = self.delim
        max_rows = max_rows or self.max_rows
        chunk_size = chunk_size or self.chunk_size
        chunk_size = chunk_size or 100
        #
        #
        with open(data_file_name, 'r') as fin:
            headers_rw = fin.readline()
            if verbose:
                print('*** headers_rw: ', headers_rw)
            headers = headers_rw[:-1].split(delim)[:-1]
            #
            # make a row-handler dictionary, until we have proper indices.
            RH = {h:k for k,h in enumerate(headers)}
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
            #
            
            n_rws = 0
            # 
            # TODO: reevaluate readlines() vs for rw in...
            n_cpu = mpp.cpu_count()
            # eventually, we might need to batch this.
            P = mpp.Pool(n_cpu)
            self.headers = headers
            results = P.map_async(self.process_row, fin, chunksize=chunk_size)
            P.close()
            data = results.get()
            #
            del results, P
            
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
            #
            #self.data=data
            # TODO: asrecarray()?
            #self.data = pandas.DataFrame(data, columns=active_headers).to_records()
            #return data
            #
            return pandas.DataFrame(data, columns=active_headers).to_records()
        #
    def process_row(self, rw):
        # use this with MPP processing:
        #
        # use this for MPP processing:
        #rws = rw.split(delim)
        return [None if vl=='' else self.types_dict.get(col,str)(vl)
                    for k,(col,vl) in enumerate(zip(self.headers, rw.split(self.delim)[:-1]))]
    #
    def active_jobs_cpu(self, n_points=100000):
        #
        #t_now = mpd.date2num(dtm.datetime.now())
        t_now = mpd.date2num( dtm.datetime.now() )
        #
        # been wrestling with datetime types, as usual, so eventually decided maybe to just
        #. use floats?
        #t_start = mpd.date2num(self.jobs_summary['Start'].astype(dtm.datetime))
        #t_start = mpd.datestr2num(numpy.datetime_as_string(self.jobs_summary['Start']))
        t_start = self.jobs_summary['Start']
        #
        t_end = self.jobs_summary['End']
        #t_end[t_end is None] = numpy.datetime64(t_now)
        t_end[t_end is None] = t_now
        #t_end = mpd.datestr2num(numpy.datetime_as_string(self.jobs_summary['End']))
        
        #t_end = self.jobs_summary['End'].astype(dtm.datetime)
        #t_end[t_end is None]=numpy.datetime64(t_now)
        #t_end = mpd.date2num(t_end)
        #
        print('** DEBUG: ', t_end.shape, t_start.shape)
        #
        #
        # Does numpy.amin() work around the None problem?
        #
        t_min = numpy.min([numpy.min(t_start), numpy.min([x for x in t_end if not x is None])])
        t_max = numpy.max([numpy.max(t_start), numpy.max([x for x in t_end if not x is None])])
        #
        # can also create X sequence like this, for minute resolution
        # X = numpy.arange(t_min, t_max, 1./(24*60))
        #
        X = numpy.linspace(t_min, t_max, 10000)
        Ns = numpy.zeros(len(X))
        Ns_cpu = numpy.zeros(len(X))
        #
        for j, t in enumerate(X):
            ix = numpy.logical_and(t_start<=t, t_end>t)
            Ns[j] = numpy.sum(ix.astype(int))
            Ns_cpu[j] = numpy.sum(self.jobs_summary['NCPUS'][ix])
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
        

class SACCT_data_from_inputs(SACCT_data_handler):
    # a SACCT_data handler that loads input file(s). Most of the time, we don't need the primary
    #. data; we only want the summary data. We'll allow for both, default to summary...
    #. assume these are in a format that will natively import as a recarray (we'll use recarray->text 
    #. methods to produce them).
    #
    def __init__(self, summary_filename=None, primary_data_filename=None):
        pass
