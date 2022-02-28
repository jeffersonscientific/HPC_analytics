# functions, classes, and other helper bits for HPC_analytics notebooks and other libs.
#
import os
import sys
import numpy
import scipy
import math
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
import hpc_lib
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
                 fig_width_tex='.8', qs=[.25, .5, .75, .9], fig_size=(10,8), SACCT_obj=None, TEX_obj=None, max_rws=None ):
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
        figs_path = os.path.join(out_path, 'figs')
        tex_path  = os.path.join(out_path, 'tex')
        self.__dict__.update({ky:vl for ky,vl in locals().items() if not ky in ('self', '__class__')})
        #
        self.HPC_tex_obj = (TEX_obj or hpc_lib.Tex_Slides(Short_title=self.Short_title,
                 Full_title=self.Full_title,
                 foutname=os.path.join(tex_path, tex_filename)) )
    #
    def cpu_hourly_activity_report(self, fout_path_name=None, qs=None, hr_am=8., hr_pm=18., fig_size=(14,15),
        n_points_wkly_hrs=None, group_name='group', SACCT_obj=None, verbose=0):
        '''
        # cpu-activity with hourly resolution. This will give us HoD, DoW distributions, cpu-activity layer cake, etc.
        '''
        fig_size = fig_size or self.fig_size
        fig_size = fig_size or (14,15)
        # handle input path:
        # default:
        #fout_path_name = fout_path_name or os.path.join(self.out_path, 'cpu_hourly_activity.png')
        fout_path_name = fout_path_name or os.path.join(self.figs_path, 'cpu_hourly_activity.png')
        # allow env-like substitution $outpath
        fout_path_name = self.fix_output_path(fout_path_name)
#        if fout_path_name.startswith('$outpath'.replace('{', '').replace('}', '')):
#            fout_path_name = fout_path_name.replace('{', '').replace('}', '')
#            fout_path_name = fout_path_name.replace('$outpath', self.out_path)
        #
        layer_cake_ave_len=7
        SACCT_obj = SACCT_obj or self.SACCT_obj
        qs = qs or self.qs
        qs = qs or numpy.array([.25, .5, .75, .9])
        #
        dtype_qs = [('time', '>i8')] + [(f'q{k+1}', '>f8') for k,x in enumerate(qs)]
        #
        ##############################
        ##############################
        #
        # Compute stuff:
        ###############################################
        #
        # CPU Activity layer cake:
        # get cpu- and jobs- activity with hourly resolution (bin_size=1./24, ie, time in units of days).
        #
        hourlies_cpu_job = hpc_lib.active_jobs_cpu(bin_size=1./24., jobs_summary=SACCT_obj.jobs_summary)
        hourlies_cpu_job['N_jobs'][numpy.isnan(hourlies_cpu_job['N_jobs'])] = 0.
        hourlies_cpu_job['N_cpu'][numpy.isnan(hourlies_cpu_job['N_cpu'])] = 0.
        #
        # we'll be splitting up into three time groups (9to5, early/after-hours, weekends)
        # NOTE: times are all local...
        # NOTE: modified the ix_a index from notebook, so adjust...
        # note: date%7==0 is a Thursday, so sat, sun = {2,3}, or we can convert to datetimes.
        ix_w   = numpy.logical_and( (hourlies_cpu_job['time'])%7 >= 2., (hourlies_cpu_job['time'])%7 <= 3. )
        ix_a   = numpy.logical_and( numpy.logical_or(hourlies_cpu_job['time']%1.<(hr_am/24.) , hourlies_cpu_job['time']%1>(hr_pm/24.)), numpy.invert(ix_w))
        ix_d   = numpy.logical_and(numpy.invert(ix_w), numpy.invert(ix_a))
        #
        #ave_bins = 7*24
        ave_bins = 24*layer_cake_ave_len
        #
        hourlies_cake_total = hpc_lib.running_mean(hourlies_cpu_job['N_cpu'], ave_bins)
        #
        # 9to5:
        X = hourlies_cpu_job['N_cpu'].copy()
        X[numpy.invert(ix_d)] = 0.
        hourlies_cake_9to5 = hpc_lib.running_mean(X,ave_bins)
        #
        # weekends:
        X = hourlies_cpu_job['N_cpu'].copy()
        X[numpy.invert(ix_w)] = 0.
        hourlies_cake_wknd = hpc_lib.running_mean(X, ave_bins)
        #
        # after-horus:
        X = hourlies_cpu_job['N_cpu'].copy()
        X[numpy.invert(ix_a)] = 0.
        hourlies_cake_afters = hpc_lib.running_mean(X, ave_bins)
        #
        if verbose:
            print('** ', numpy.sum(hourlies_cpu_job['N_cpu']), numpy.sum(hourlies_cpu_job['N_cpu'][ix_d]),
              numpy.sum(hourlies_cpu_job['N_cpu'][ix_a]), numpy.sum(hourlies_cpu_job['N_cpu'][ix_w]))
        #
        ###############################
        #
        # Quantiles:
        #
        dtype_qs = [('time', '>i8')] + [(f'q{k+1}', '>f8') for k,x in enumerate(qs)]
        periodic_hourly_weekdays_cpus = numpy.zeros((24,), dtype=dtype_qs)
        periodic_hourly_weekdays_cpus['time'] = numpy.arange(24)
        #
        if verbose:
            print('** dtype: ', dtype_qs)
            print('** ', periodic_hourly_weekdays_cpus)
            print('** ', periodic_hourly_weekdays_cpus['time'])
        #
        for k in periodic_hourly_weekdays_cpus['time']:
            q_vals = numpy.nanquantile(hourlies_cpu_job['N_cpu'][numpy.logical_and(numpy.invert(ix_w),
                                            (24*(hourlies_cpu_job['time']%1)).astype(int)==k)], qs)
            #
            for j,q in enumerate(q_vals):
                periodic_hourly_weekdays_cpus[f'q{j+1}'][k]=q
        #
        if verbose:
            print('** **\n', periodic_hourly_weekdays_cpus)
        #
        periodic_daily_925_cpus = numpy.zeros((7,), dtype=dtype_qs)
        periodic_daily_cpus = numpy.zeros((7,), dtype=dtype_qs)
        periodic_daily_925_cpus['time'] = numpy.arange(7)
        periodic_daily_cpus['time'] = numpy.arange(7)
        #
        for k in periodic_daily_cpus['time']:
            ix_925 = numpy.logical_and(numpy.logical_and((hourlies_cpu_job['time']%1. >= hr_am/24.),
                                                         (hourlies_cpu_job['time']%1 <= hr_pm/24.)),
                                       (((hourlies_cpu_job['time']+3)%7).astype(int)==k)
                                      )
            q_vals_925 = numpy.nanquantile(hourlies_cpu_job['N_cpu'][ix_925], qs)
            q_vals     = numpy.nanquantile(hourlies_cpu_job['N_cpu'][(((hourlies_cpu_job['time']+3)%7)).astype(int)==k], qs)
            #
            print('*** DEBUG: q_vals', q_vals)
            for j,q in enumerate(q_vals):
                periodic_daily_cpus[f'q{j+1}'][k]=q
            for j,q in enumerate(q_vals_925):
                periodic_daily_925_cpus[f'q{j+1}'][k]=q
        #
        ####################################
        #
        # set up figure and axes, then plot
        fg = plt.figure(figsize=fig_size)
        n_cols = 2
        n_rws  = 4
        ax1 = plt.subplot(n_rws,1,1)
        ax2 = plt.subplot(n_rws,1,2, sharex=ax1)
        ax3 = plt.subplot(n_rws,n_cols,5)
        ax4 = plt.subplot(n_rws,n_cols,6)
        ax5 = plt.subplot(n_rws,n_cols,7, projection='polar')
        ax6 = plt.subplot(n_rws,n_cols,8, projection='polar')
        #
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()
        ax6.grid()
        #
        fg.suptitle('CPU Usage Report', size=16)
        ax1.set_title('Active CPUs')
        ax2.set_title(f'Active CPUs layer cake ({layer_cake_ave_len} day average)')
        ax3.set_title('CPUs Active: Hourly, weekdays')
        ax4.set_title('CPUs Active: DoW, 9-5')
        #
        # ix=numpy.logical_and( T%1.>.3 , T%1<.8, (T-4)%7)
        # ax.plot(hourlies['time'], hourlies['cpu_hours'])
        # ax.plot(hourlies['time'][ix], hourlies['cpu_hours'][ix], ls='', marker='.')
        ax1.plot(hourlies_cpu_job['time'], hourlies_cpu_job['N_cpu'], color='b', lw=1.)
        #ax1.fill_between(hourlies_cpu_job['time'], 0., hourlies_cpu_job['N_cpu'], color='b', alpha=.2, zorder=1)
        X = hourlies_cpu_job['N_cpu'].copy()
        X[numpy.invert(ix_d)]=0.
        ax1.fill_between(hourlies_cpu_job['time'], 0., X, alpha=.2)
        #
        X = hourlies_cpu_job['N_cpu'].copy()
        X[numpy.invert(ix_w)]=0.
        ax1.fill_between(hourlies_cpu_job['time'], 0., X, alpha=.2)
        #
        X = hourlies_cpu_job['N_cpu'].copy()
        X[numpy.invert(ix_a)]=0.
        X[ix_w]=0.
        ax1.fill_between(hourlies_cpu_job['time'], 0., X, alpha=.2)
        #

        # ax1.plot(hourlies_cpu_job['time'][ix_d], hourlies_cpu_job['N_cpu'][ix_d], ls='', marker='.')
        # ax1.plot(hourlies_cpu_job['time'][ix_w], hourlies_cpu_job['N_cpu'][ix_w], ls='', marker='.')
        # ax1.plot(hourlies_cpu_job['time'][ix_a], hourlies_cpu_job['N_cpu'][ix_a], ls='', marker='.')
        #
        T = hourlies_cpu_job['time']
        ln, = ax2.plot(T[ave_bins-1:], hourlies_cake_total, ls='-', color='r', label='total')
        #ax2.plot(T, hourlies['cpu_hours'], ls='-', marker='')
        X = hourlies_cake_wknd.copy()
        ln, = ax2.plot(T[ave_bins-1:], X, ls='--', label='weekends')
        clr = ln.get_color()
        ax2.fill_between(T[ave_bins-1:], 0., X, color=clr, alpha=.2 )
        #
        X0 = X.copy()
        X += hourlies_cake_afters
        ln, = ax2.plot(T[ave_bins-1:], X, ls='--', label='after-hours')
        clr = ln.get_color()
        ax2.fill_between(T[ave_bins-1:], X0, X, color=clr, alpha=.2 )
        #
        X0 = X.copy()
        X += hourlies_cake_9to5
        ln, = ax2.plot(T[ave_bins-1:], X, ls='--', marker='', label='9to5')
        clr = ln.get_color()
        ax2.fill_between(T[ave_bins-1:], X0, X, color=clr, alpha=.2 )
        ax2.legend(loc=0)
        # NOTE: see hpc_lib.make_report(). This sometimes fails with negative or 0 valued dates (??) If it's a problem, it can be
        #   handled using a max() function.
        # this may need to go somewhere. not sure this is where...
        fg.canvas.draw()
        print('** ** DEBUG: ', [s.get_text() for s in ax2.get_xticklabels()])
        lbls = [hpc_lib.simple_date_string(mpd.num2date(max(1, float(s.get_text())) ) ) for s in ax2.get_xticklabels()]
        ax2.set_xticklabels(lbls)
        #
        #
        ln, = ax3.plot(periodic_hourly_weekdays_cpus['time'], periodic_hourly_weekdays_cpus['q2'], lw=3)
        clr = ln.get_color()
        ax3.fill_between(periodic_hourly_weekdays_cpus['time'], periodic_hourly_weekdays_cpus['q2'],
                         periodic_hourly_weekdays_cpus['q3'], color=clr, alpha=.15 )
        ax3.fill_between(periodic_hourly_weekdays_cpus['time'], periodic_hourly_weekdays_cpus['q2'],
                         periodic_hourly_weekdays_cpus['q4'], color=clr, alpha=.15 )
        #
        xx = numpy.max(numpy.append([1000], periodic_hourly_weekdays_cpus['q1']))
        ax3.set_ylim(xx, 5000)
        #
        X = periodic_daily_925_cpus.copy()
        ax4.plot(X['time'], X['q2'], lw=3, color='b', label='9to5')
        ax4.fill_between(X['time'], X['q2'], X['q3'], color='b', alpha=.15 )
        ax4.fill_between(X['time'], X['q1'], X['q4'], color='b', alpha=.15 )
        X = periodic_daily_cpus
        ax4.plot(X['time'], X['q1'], lw=3, color='m', ls='--', label='all')
        ax4.fill_between(X['time'], X['q2'], X['q4'], color='m', alpha=.08)
        ax4.legend(loc=0)
        #
        ax5.set_theta_direction(-1)
        ax5.set_theta_offset(math.pi/2.0)
        X = periodic_hourly_weekdays_cpus['time'].copy()*scipy.constants.pi*2.0/float(len(periodic_hourly_weekdays_cpus))
        ln, = ax5.plot(X, periodic_hourly_weekdays_cpus['q2'], lw=3, marker='.')
        clr = ln.get_color()
        ax5.plot(numpy.linspace(0., math.pi*2., 100), numpy.ones(100)*numpy.mean(periodic_hourly_weekdays_cpus['q2']),
                 ls='--', lw=2., color=clr, alpha=.7, label='mean, 50th' )
        ax5.set_xticks(numpy.arange(0., math.pi*2., math.pi*2/24.))
        ax5.set_xticklabels(numpy.arange(0,24, 1.))
        y_min = .5*(numpy.max(periodic_hourly_weekdays_cpus['q1'])+numpy.min(periodic_hourly_weekdays_cpus['q2']))
        ax5.set_ylim(y_min, )
        ax5.grid()
        ax5.legend(loc='upper right')
        #
        # TODO: figure out phase of DoW circle-plot.
        ax6.set_theta_direction(-1)
        ax6.set_theta_offset(math.pi/2.0)
        X = periodic_daily_925_cpus.copy()
        ln, = ax6.plot(X['time']*math.pi*2.0/float(len(X)+1.), X['q2'], lw=3, marker='o', label='9to5')
        clr = ln.get_color()
        ax6.plot(numpy.linspace(0., math.pi*2., 100), numpy.ones(100)*numpy.mean(X['q2']), ls='--', lw=2.,
                 color=clr, alpha=.7 , label='mean, 50th')
        ax6.set_xticks=numpy.arange(0., math.pi*2., math.pi*2./7.)
        ax6.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        y_min = .5*(numpy.max(X['q1']) + numpy.min(X['q2']))
        ax6.set_ylim(y_min, )
        ax6.grid()
        ax6.legend(loc='upper left')
        #
        # Save figure:
        plt.savefig(fout_path_name)
        #
        return fout_path_name
    #
    def wait_stats_report(self, fout_path_name=None, qs=None, fig_size=(14,12), group_name='group', SACCT_obj=None, verbose=0):
        '''
        # cpu-activity with hourly resolution. This will give us HoD, DoW distributions, cpu-activity layer cake, etc.
        '''
        fig_size = fig_size or self.fig_size
        fig_size = fig_size or (14,12)
        #
        #fout_path_name = fout_path_name or os.path.join(self.out_path, 'wait_times_stats.png')
        fout_path_name = fout_path_name or os.path.join(self.figs_path, 'wait_times_stats.png')
        # allow env-like substitution $outpath
        fout_path_name = self.fix_output_path(fout_path_name)
#        if fout_path_name.startswith('$outpath'.replace('{', '').replace('}', '')):
#            fout_path_name = fout_path_name.replace('{', '').replace('}', '')
#            fout_path_name = fout_path_name.replace('$outpath', self.out_path)
        #
        SACCT_obj = SACCT_obj or self.SACCT_obj
        #
        # NOTE: during developmebnt, we used qs_ps to refer the the q-percentiles, not the quantile values themselves.
        #  we'll maintain external (to this function) consistency by stickign with the qs notation, but use qs_ps
        #  internally, mostly so we can cut-n-paste from dev. It might be desirable to eventually get rid of the qs_ps notation entirely
        qs_ps = qs or self.qs
        qs_ps = qs_ps or numpy.array([.25, .5, .75, .9])
        #
        # end handle inputs
        #
        #############
        #dtype_qs = [('time', '>i8')] + [(f'q{k+1}', '>f8') for k,x in enumerate(qs)]
        wait_times = 60.*24.*(SACCT_obj.jobs_summary['Start'] - SACCT_obj.jobs_summary['Submit'])
        wait_time_units='minutes'
        #
        wait_times_per_dow = hpc_lib.day_of_week_distribution(time=SACCT_obj.jobs_summary['Submit'], Y=wait_times)
        wait_times_per_hod = hpc_lib.hour_of_day_distribution(time=SACCT_obj.jobs_summary['Submit'], Y=wait_times)
        #
        qs_waittimes = numpy.nanquantile(wait_times,qs_ps)
        print('*** Wait-time quantiles: {}'.format(qs_waittimes))
        ix_q4 = numpy.logical_and(numpy.isnan(wait_times)==False, wait_times<qs_waittimes[-1])
        #
        fg = plt.figure(figsize=fig_size)
        ax1a = plt.subplot(3,2,1, projection='polar')
        ax1a.set_theta_direction(-1)
        ax1a.set_theta_offset(math.pi/2.0)
        ax1a.set_title('Wait-Time DoW quantiles (hours)')
        #
        ax1b = plt.subplot(3,2,2, projection='polar')
        ax1b.set_theta_direction(-1)
        ax1b.set_theta_offset(math.pi/2.0)
        ax1b.set_title('Wait-Time ToD quantiles (hours)')
        #
        ax2a = plt.subplot(3,2,3)
        ax2a.set_title('Wait-Time DoW (clock plot)')
        ax2b = plt.subplot(3,2,4)
        ax2b.set_title('Wait-Time ToD (clock plit)')
        #
        ax3 = plt.subplot(3,2,5)
        ax4 = plt.subplot(3,2,6)
        #
        X_dow = wait_times_per_dow['DoW']*math.pi*2./7.
        ln, = ax1a.plot(X_dow, wait_times_per_dow['q2'], ls='-', marker='o')
        clr = ln.get_color()
        ax1a.fill_between(X_dow, wait_times_per_dow['q2'], wait_times_per_dow['q3'], color=clr, alpha=.2)
        ax1a.set_xticks(X_dow)
        ax1a.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        #
        X_hod = wait_times_per_hod['hour']*math.pi*2./float(len(wait_times_per_hod))
        ln, = ax1b.plot(X_hod, wait_times_per_hod['q2'], ls='-', marker='o')
        clr = ln.get_color()
        ax1b.fill_between(X_hod, wait_times_per_hod['q2'], wait_times_per_hod['q3'], color=clr, alpha=.2)
        ax1b.set_xticks(X_hod )
        ax1b.set_xticklabels(numpy.arange(0,24,1))
        #
        X_dow = wait_times_per_dow['DoW']
        ln, = ax2a.plot(X_dow, wait_times_per_dow['q2'], ls='-', marker='o')
        clr = ln.get_color()
        ax2a.fill_between(X_dow, wait_times_per_dow['q2'], wait_times_per_dow['q3'], color=clr, alpha=.2)
        ax2a.set_xticks(numpy.arange(0, 7) )
        ax2a.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax2a.set_ylabel(wait_time_units.capitalize())
        #
        X_hod = wait_times_per_hod['hour']
        ln, = ax2b.plot(X_hod, wait_times_per_hod['q2'], ls='-', marker='o')
        clr = ln.get_color()
        ax2b.fill_between(X_hod, wait_times_per_hod['q2'], wait_times_per_hod['q3'], color=clr, alpha=.2)
        #
        h1 = ax3.hist(wait_times[ix_q4], bins=1000, density=True)
        #ax3.set_xlim(0., 1.)
        ax3.set_title(f'Wait time PDF of {qs_ps[-1]:.2f} quantile')
        ax3.set_xlabel(wait_time_units.capitalize())
        ax3.set_ylabel('Percent')
        #
        h2 = ax4.hist(wait_times[ix_q4], bins=100, cumulative=True, density=True, histtype='step')
        h2 = ax4.hist(wait_times[ix_q4], bins=100, cumulative=True, density=True, histtype='stepfilled', alpha=.2)
        #ax4.set_xlim(0, 1.)
        ax4.set_title(f'Wait time CDF of {qs_ps[-1]:.2f} quantile')
        ax4.set_xlabel(wait_time_units.capitalize())
        ax4.set_ylabel('Percent')
        #
        for ax in (ax2a, ax2b, ax3, ax4):
            ax.grid()
        # Save figure:
        plt.savefig(fout_path_name)
        #
        # TODO: consider returning a standard class or just a dict. with meta-data
        return fout_path_name
    #
    def fix_output_path(self, out_path):
        '''
        # standard script to process output_path, to permit wildcards or other tricks and easter-eggs
        '''
        if out_path.startswith('$outpath'.replace('{', '').replace('}', '')):
            out_path = out_path.replace('{', '').replace('}', '')
            out_path = out_path.replace('$outpath', self.out_path)
        #
        return out_path


        
    
