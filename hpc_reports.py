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
#
class SACCT_report_handler(object):
    '''
    # Semi-versatile reports handler.
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
