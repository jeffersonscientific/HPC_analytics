# functions, classes, and other helper bits for HPC_analytics notebooks and other libs.
#
import numpy
import scipy
#import matplotlib
import matplotlib.dates as mpd
#import pylab as plt
import datetime as dtm
#
#import pandas
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