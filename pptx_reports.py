###
# Generate PPTX sacct-partition report
###

import numpy
import scipy
import math
import matplotlib as mpl
import matplotlib.dates as mpd
import pylab as plt
import datetime as dtm
import pytz
import multiprocessing as mpp
import pickle
import sys
import os
#
import argparse
import dataclasses
#
import pptx
from pptx.enum.text import PP_ALIGN
#
#import HPC_analytics.hpc_lib as hpc_lib
#import hpc_lib
# use this syntax to force the path-relative import. If we start doing a pip install, we might
#. be able to/need to change this to the regular "import hpc_lib" syntax
# which of these works seems to depend on how we load it -- eg, batch job vs notebook
try:
    from . import hpc_lib
except:
    import hpc_lib

GB=1024.**3.
#
# def run_partition_pptx(partition=None, group=None, user=None, 
#                           start_dtm=None, end_dtm=None, report_len_days=None, n_cpus=None,
#                           delim_sacct=None, sacct_data_path=None,
#                           report_path='pptx_reports', report_name=None
#                       ):
class SACCT_PPTX_Deck(object):
    #
    # How much can we abstract this? We can definitly define each slide type. Can we abstract/generalize the pie-charts, etc?
    # Let's keep this simpler for now. If we abstract, we'll use .yaml to define reports. .yaml is a cleaner way to define
    # report details. So we'll leacve 'ageanda': here for now, but we won't use it.
    # 
    slide_types_dict = {'title': {'f': 'self.title_slide'}, 'agenda':{'f': ''},
                        'active_cpus':{'f':'pass', 'agenda_text': 'Active CPUs and Jobs'},
                        'active_cpus_breakout':{'f':'function', 'agenda_text':'Active CPUs Breakout', 'kwargs':{'group_by':'account'}},
                        'job_topologies':{'agenda_text': 'Job Parallel Topologies'},
                        'activite_gpus':{'agenda_text': 'Active GPUs and Jobs'}, 
                        'active_gpu_breakout':{'f':'function', 'kwargs':{'group_by':'account'}}, 
                        'gpu_efficiency':{}
                                 }
                        
    slides_list_default = ['title', 'agenda', 'active_cpus', 'job_topologies']
    #
    def __init__(self, partition=None, group=None, user=None, 
                          start_dtm=None, end_dtm=None, report_len_days=None, n_cpus=None,
                          delim_sacct=None, sacct_data_path=None,
                          report_path='pptx_reports', report_name=None, slides_list=None,
                          extra_slide_types=None, cluster_name="Sherlock", report_type_description=None,
                          delete_existing_report=True, verbose=False
                      ):

        '''
        Add some doc strings here...
        '''
#         # Handle some default values, too bulky to just use ={default}
#         self.slide_types_dict.update(extra_slide_types)
        #
        if slides_list is None:
            slides_list = self.slides_list_default.copy()
        slide_types = {ky: self.slide_types_dict[ky] for ky in slides_list}
        #  
        # Collect data, compute metrics, create figures and package into a ppxt power point report.
        #. Note: nominally/usually, group= and user= should be None, but we reserve the right to run
        #. a report for a group or user within a report. Some of the metrics/plots might not make much sense --
        #  -- eg, for "CPU Hours by group" , but there are cases where this option will add value.
        #
        # get SACCT_ and SINFO_obj objects:
        SINFO_obj = hpc_lib.SINFO_obj(partition=partition, sinfo_prams={'--Node':''}, Format_fields='node',
                            verbose=False, do_sinfo=True
                            )
        
        print(f'** DEBUG: {type(start_dtm)}, {type(end_dtm)}')
        SACCT_obj = hpc_lib.get_sacct_direct_or_hdf5(partition=partition, group=group, user=user, 
                               start_dtm=start_dtm, end_dtm=end_dtm, report_len_days=report_len_days,
                                n_cpu=n_cpus,
                                delim_sacct=delim_sacct, sacct_data_path=sacct_data_path
                                )
        #
        # It is not clear yet how much value there is to break this up into functions. Each report is a little different,
        #. so we will need to create some figures, etc. from scratch each time -- even ones that look very similar. That said,
        #  there is opportunity to reuse code, for example (pseudo-code), 
        #.  make_pie_of_cpu_hours(vals=SACCT['elapsed']*sacct['ncpus'], group_by={report_type}).
        #. On the other hand, Available CPUs/GPUs is only meaningful for a partition report.
        #
        # Partition specs:
        n_cpus_partition = SINFO_obj.total_cpus
        n_gpus_partition = SINFO_obj.total_gpus
        n_cpu_cpus_partition = SINFO_obj.total_cpu_cpus
        n_gpu_cpus_partition = SINFO_obj.total_gpu_cpus
        #
        # report path:
        if report_path is None:
            report_path = 'pptx_reports'
        if report_name is None or report_name.replace(' ','')=='':
            # TODO: clean up the date strings. Take dates from SACCT_obj?
            report_name = f'p{str(partition).upper()}_g{str(group).upper()}_u{str(user).upper()}_{str(start_dtm)}_{str(end_dtm)}.ppt'
        report_name = report_name.replace(' ', '_').replace(':', '_')
        report_pathname = os.path.join(report_path, report_name)
        img_path        = os.path.join(report_path, f'images_{report_name}')
        #
        if not os.path.isdir(img_path):
            os.makedirs(img_path)
        #
        if os.path.exists(report_pathname) and delete_existing_report:
            os.remove(report_pathname)
        #
        if verbose:
            print(f'** CPUs: {n_cpus_partition}, GPUs: {n_gpus_partition}, cup-CPUs: {n_cpu_cpus_partition}, gpu_CPUs:{n_gpu_cpus_partition}')
            print(f'** Report Path: {report_pathname}')
        #
        ####
        #
        # We should now have the data objects we need to compute metrics, make figures, create report.
        #
        # CREATE a Presentation() object:
        pptx_report = pptx.Presentation()
        pptx_report.save(report_pathname)
        #
        # Don't need these as return dataclasses, since we're in a Class instance now.
        # @dataclasses.dataclass
        # class REP:
        #     sinfo: hpc_lib.SINFO_obj
        #     sacct: type(SACCT_obj)
        # #return REP(SINFO_obj, SACCT_obj)
        # REP = REP(SINFO_obj, SACCT_obj)
        #
        self.__dict__.update({ky: vl for ky, vl in locals().items() if ky not in ['self', '__class__'] } )
        #
        #######
        #t_slide = self.title_slide()
        #a_slide = self.agenda_slide()
    def reset_report(self):
        report_pathname = self.report_pathname
        #
        if os.path.exists(report_pathname):
            os.remove(report_pathname)
        pptx_report = pptx.Presentation()
        pptx_report.save(report_pathname)
        #
        return None
    #
    def save(self, filepath=None):
        filepath = (filepath or self.report_pathname)
        self.pptx_report.save(filepath)
    #
    def main(self, verbose=False):
        '''
        Build report pages.
        '''
        verbose = verbose or self.verbose
        #
        if verbose:
            print(f'*** VERBOSE: Begin building report: {self.report_pathname}')
        #
        self.reset_report()
        t_s          = self.slide_title()
    def slide_title(self, do_save=False):
        '''
        Base class title slide.
        '''
        # TITLE SLIDE:
        prs = self.pptx_report
        #partition = self.partition
        SACCT_obj = self.SACCT_obj
        #
        slide = prs.slides.add_slide(prs.slide_layouts[0])

        slide.shapes.title.text = f"SACCT Reporting for {self.cluster_name}\
         Partition: {SACCT_obj.partition}, Group: {SACCT_obj.account}, User: {SACCT_obj.user}"
        slide.placeholders[1].text = f'({str(SACCT_obj.start_date.date())} - {str(SACCT_obj.end_date.date())})'
        #
        if do_save:
            prs.save(self.report_pathname)

    
    # Slides that can be easily abstracted can go here:
    def slide_cpus_pie_and_cdf(self, SACCT_obj=None, SINFO_obj=None,
         group_by='Account', cpus_col='NCPUS', fig_fname=None, group_data=None, sum_data=None,
         verbose=False, do_save=False):
         #, report_type_description=None):
        '''
        # TODO: this function, at least the compute part, can probably be abstracted into
        #. the parent Class(), with group_by and one or two other prams.
        # Most active groups (presumably in a partition)
        # pie chart and CDF.
        # @ SACCT_obj, SINFO_obj: SACCT_ and SINFO_obj objects.
        # @group_by, @cpus_col [str]: cheap way to specify group_by or cpus col.
            Use these cols (by label) to compute figures.
        # @fig_name: override defautl figure naming convention (might be useful or necessary
        #.  if we abstract.)
        # @group_data, @sum_data: override the data columns the hard way. Provide explicit data, eg.
        # group_data = my_SACCT_obj['Elapsed']*my_SACCT_obj['NCPUS']*24.
        '''
        verbose = verbose or self.verbose
        #report_type_description = report_type_description or self.report_type_description
        #
        SACCT_obj = SACCT_obj or self.SACCT_obj
        SINFO_obj = SINFO_obj or self.SINFO_obj
        #
        if group_by is None and not group_data is None:
            group_by = 'group'
        if group_by is None and group_data is None:
            group_by = 'Account'
        #
        if sum_data is None:
            sum_data = SACCT_obj['Elapsed']*SACCT_obj[cpus_col]*24.
        if group_data is None:
            group_data = SACCT_obj[group_by]
        #
        if fig_fname is None:
            fig_fname = f'active_pie_and_cdf_by_{cpus_col}_{group_by}.jpg'
        img_path = self.img_path
        ###
        ### Compute main bits first:
        # NOTE: Raw SACCT time elapsed data in days...
        pi_cpuh = hpc_lib.get_pie_slices(sum_data=sum_data, slice_data=group_data)
        pi_cpuh_lbls = pi_cpuh['name'].astype(str)
        pi_cpuh_vls  = pi_cpuh['value'].astype(str)

        fg = plt.figure(figsize=(8,16))
        ax1 = fg.add_subplot(2,1,1)
        ax2 = fg.add_subplot(2,1,2)
        #
        ix_picpuh = numpy.argsort(pi_cpuh['value'])
        X = pi_cpuh['value'][ix_picpuh]
        Y = numpy.arange(1, len(X)+1)
        #
        # TODO: enable normalization (report pct, not raw numbers). Enable count or pct? both? either?
        NORM = 1.
        NORM = float(Y[-1])
        y_lbl = 'Percentile'
        if NORM == 1.0:
            y_lbl = 'Count'
        #
        ax1.pie(pi_cpuh['value'], labels=pi_cpuh_lbls)
        ax2.plot(X, Y/NORM, ls='-', marker='.')
        #
        qs = [.5, .75, .9,.95]
        Qs = numpy.quantile(X,qs)
        #
        for x,q in zip(Qs,qs):
            k = numpy.searchsorted(X,x)
            y = ((x-X[k-1])*Y[k-1] + (X[k]-x)*Y[k])/(X[k] - X[k-1])
            ax2.plot([x,x,0], numpy.array([0.,y,y])/NORM, label=f'q{int(q*100)}: {x:,.2f}')
        #for (x,y),nm in zip([q_50, q_90], ['q50', 'q90']):
        #    ax2.plot([x,x,0], numpy.array([0,y,y])/NORM, label=f'{nm}:{x:.2f}')
        #
        ax2.grid()
        ax2.set_xlabel('CPU hours by {group_by or "group"}', size=16)
        ax2.set_ylabel(y_lbl, size=16)
        ax2.legend(loc=0)
        #
        # if we abstract, we'll need to handle the naming convention:
        fg_path_group_cpuh_pie_and_cdf = os.path.join(img_path, fig_fname)
        fg.savefig(fg_path_group_cpuh_pie_and_cdf)
        #
        if verbose:
            print('** ', X[-2:])
            print('** mean,median: ', numpy.mean(pi_cpuh['value']), numpy.median(pi_cpuh['value']))
            print('** quantiles [.5, .9, .99])', numpy.quantile(pi_cpuh['value'], [.5, .9, .99]))
            print('** ', numpy.array(sorted(pi_cpuh['value']))[-5:])
        #
        ###
        # Make the slide:
        #
        prs = self.pptx_report
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        #slide = prs.slides.add_slide(prs.slide_layouts[8])

        slide.shapes.title.text = f"Most Active\nby {group_by}: {cpus_col}"
        #slide.shapes.title.alignment = PP_ALIGN.LEFT
        for pp in slide.shapes.placeholders[0].text_frame.paragraphs:
            #slide.shapes.placeholders[0].text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
            pp.alignment = PP_ALIGN.LEFT
        #
        # figure:
        left = pptx.util.Inches(6.)
        top = pptx.util.Inches(2)
        top_pic = pptx.util.Inches(.25)
        height = pptx.util.Inches(4)
        height_pic = pptx.util.Inches(7)
        pic = slide.shapes.add_picture(fg_path_group_cpuh_pie_and_cdf, left, top_pic, height=height_pic)
        # 
        # use remove, insert to move the pie charts image to the back.
        # this works, but triggers "PowerPoint must repair this... message and action."
        # slide.shapes._spTree.remove(pic._element)
        # slide.shapes._spTree.insert(2, pic._element)
        #
        # text/bullets:
        #bullets_text = slide.shapes.add_textbox(left=pptx.util.Inches(.5), top=top, width=pptx.util.Inches(2.5),
        #                                   height=pptx.util.Inches(5))
        bullets_text = slide.shapes.placeholders[1]
        bullets_text.left=pptx.util.Inches(.5)
        bullets_text.width=pptx.util.Inches(4.5)
        bullets_text.top=top
        #
        tf = bullets_text.text_frame

        k_max=5
        font_pt=pptx.util.Pt(18)
        #p = tf.add_paragraph()
        p = tf.paragraphs[0]
        p.text = f"Top {k_max} active {group_by}s"
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        p.font.bold = True
        #p.bullet.type = pptx.enum.text.MSO_BULLET_SYMBOL_NONE

        ix_cpuh = numpy.argsort(pi_cpuh['value'])[::-1]
        total_cpuh = numpy.sum(pi_cpuh['value'])
        for k, (cpuh,nm) in enumerate(zip(pi_cpuh['value'][ix_cpuh],pi_cpuh['name'][ix_cpuh])):
            if hasattr(nm, 'decode'):
                nm = nm.decode()
            print(f'**[{k}] {nm}: {cpuh}')
            if k>=k_max: break
            #
            p = tf.add_paragraph()
            p.text = f"{nm}: {cpuh:,.2f} cpu-hrs ({100.*cpuh/total_cpuh:.2f}%)"
            p.alignment = PP_ALIGN.LEFT
            p.level = 1

        # Quantiles:
        p.text = f"Quantiles:"
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        p.font.bold = True
        #
        n_max = 4
        for j, (x,q) in enumerate(zip(Qs[::-1],qs[::-1])):
            if j>=n_max:
                break
            print('** **: ', x,q)
            k = numpy.searchsorted(X,x)
            y = ((x-X[k-1])*Y[k-1] + (X[k]-x)*Y[k])/(X[k] - X[k-1])
            #ax2.plot([x,x,0], numpy.array([0.,y,y])/NORM, label=f'q{int(q*100)}: {x:,.2f}')
            p = tf.add_paragraph()
            p.text = f'q{int(q*100)}: {x:,.2f} CPUh'
            p.alignment = PP_ALIGN.LEFT
            p.level = 1
            
        for p in tf.paragraphs[1:]:
            p.runs[0].font.size = font_pt
        if do_save:
            self.save()
        #
    def slide_jobs_topologies(self, SACCT_obj=None, img_path=None, img_name=None,
     verbose=False, do_save=False):
        #,cpus_col='NCPUS', CPU_HOURS_data=None):
        '''
        # Job topologies figures, eg. "distribution of CPU-hours using 1,2,N tasks"
        # TODO: add cpus_col and/or CPU_HOURS_data to better customize this report. This will 
        #.  the same require changes to hpc_lib.py.report_topologies_cpu_compute().
        #.  For now, we can pass SACCT_obj.
        #
        #@ cpus_col: quick hack of a way to switch to cpus_col='NGPUs'. For anything more fancy,
        # substitute SACCT_obj, or submit an entire CPU_HOURS_data vector.
        '''
        SACCT_obj = SACCT_obj or self.SACCT_obj
        
        img_path = img_path or self.img_path
        if img_name is None:
            img_name = 'jobs_topology_cdfs.jpg'
        verbose = verbose or self.verbose
        total_cpu_hours = numpy.sum(SACCT_obj['Elapsed']*SACCT_obj['NCPUS'])*24.
        #
        # TODO: Add discriminators to separate GPU and CPU jobs? We can do this pretty easily
        #. by providing a SACCT_obj[ix].
        fg_topology = SACCT_obj.report_topologies_cpu_compute(SACCT_obj=SACCT_obj)
        img_path_topology = os.path.join(img_path, img_name)
        fg_topology.fig.savefig(img_path_topology)
        #
        # Make the slide:
        prs = self.pptx_report
        #
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        #slide = prs.slides.add_slide(prs.slide_layouts[8])
        #
        #slide.shapes.title.text = f"{partition.upper()} CPU Hours, CDFs by topology"
        slide.shapes.title.text = f"CPU Hours, CDFs by topology"
        #slide.shapes.title.alignment = PP_ALIGN.LEFT
        for pp in slide.shapes.placeholders[0].text_frame.paragraphs:
            #slide.shapes.placeholders[0].text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
            pp.alignment = PP_ALIGN.LEFT
        #
        # text:
        left_text   = pptx.util.Inches(.5)
        top_text    = pptx.util.Inches(2)
        height_text = pptx.util.Inches(6)
        width_text  = pptx.util.Inches(3)
        #
        # figure:
        left_pic   = pptx.util.Inches(3.25)
        top_pic    = pptx.util.Inches(2)
        height_pic = pptx.util.Inches(4)
        width_pic  = pptx.util.Inches(6.75)

        pic = slide.shapes.add_picture(os.path.join(img_path_topology), left=left_pic,
                                    top=top_pic, width=width_pic, height=height_pic)
        # 
        # grab the existing bullets-formatted text box:
        bullets_text       = slide.shapes.placeholders[1]
        bullets_text.left  = left_text
        bullets_text.width = width_text
        bullets_text.top   = top_text
        tf = bullets_text.text_frame

        font_pt=pptx.util.Pt(16)
        #p = tf.add_paragraph()
        p = tf.paragraphs[0]
        # {SACCT_obj.partition.upper()}
        p.text = f"{100*fg_topology.quantiles[-1]}% of CPU-hours use less than:"
        p.bullet = False

        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        p.font.bold = True
        #p.bullet.type = pptx.enum.text.MSO_BULLET_SYMBOL_NONE
        #
        for k, (vl,nm) in enumerate([(fg_topology.quantiles_ncpus[-1], 'CPUs'),
                                    (fg_topology.quantiles_nnodes[-1], 'Nodes'),
                                    (fg_topology.quantiles_ntasks[-1], 'Tasks')]):
            #
            if verbose:
                print('** vl: ', vl)
            #
            p = tf.add_paragraph()
            p.text = f"{vl:.1f}: {nm}"
            p.alignment = PP_ALIGN.LEFT
            p.level = 1

        for nm,vl in [('CPU', fg_topology.cpuh_one_NCPUS), ('Task', fg_topology.cpuh_one_NTASKS),
                    ('Node', fg_topology.cpuh_one_NNODES)]:
            p = tf.add_paragraph()
            pct = 100.*vl/total_cpu_hours
            p.text = f'One {nm}: {pct:.2f}%'
            p.alignment = PP_ALIGN.LEFT
            p.level = 0

        for p in tf.paragraphs[1:]:
            p.runs[0].font.size = font_pt
        if do_save:
            self.save()

class SACCT_Partition_PPTX_Deck(SACCT_PPTX_Deck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_type = "partition"

    def main(self, verbose=False):
        '''
        Build report pages.
        '''
        verbose = verbose or self.verbose
        #
        if verbose:
            print(f'*** VERBOSE: Begin building report: {self.report_pathname}')
        #
        self.reset_report()
        t_s          = self.slide_title()
        a_a          = self.slide_agenda()
        cpu1         = self.slide_active_cpus()
        cpu_2        = self.slide_active_cpus(gpu_jobs=False)
        pie_cdf_cpus = self.slide_cpus_pie_and_cdf(SACCT_obj=None, SINFO_obj=None,
                group_by='Account', cpus_col='NCPUS', fig_fname=None, group_data=None, sum_data=None)
        topos        = self.slide_jobs_topologies()
        #
        # GPUS
        # TODO: Add a "GPUs" intra-deck title slide.
        gpus_activity = self.slide_active_cpus(cpu_jobs=False, gpu_jobs=True, cpu_col='NGPUs')
        pie_cdf_gpus = self.slide_cpus_pie_and_cdf(SACCT_obj=None, SINFO_obj=None,
                group_by='Account', cpus_col='NGPUs', fig_fname=None, group_data=None, sum_data=None)
        self.save()

    def slide_title(self, do_save=False):
        # TITLE SLIDE:
        prs = self.pptx_report
        #partition = self.partition
        SACCT_obj = self.SACCT_obj
        #
        slide = prs.slides.add_slide(prs.slide_layouts[0])

        slide.shapes.title.text = f"SACCT Reporting for {self.cluster_name} Partition: {SACCT_obj.partition}"
        slide.placeholders[1].text = f'({str(SACCT_obj.start_date.date())} - {str(SACCT_obj.end_date.date())})'
        # TODO: If user= or group= or account= are included, not those here? Add a special slide to record
        #. detailed runtime parameters?

        if do_save:
            prs.save(self.report_pathname)
        
    def slide_agenda(self, do_save=False):
        prs = self.pptx_report
        # 
        bullet_layout = prs.slide_layouts[1]
        overview_slide  = prs.slides.add_slide(bullet_layout)
        shapes = overview_slide.shapes
        #
        # equivalent:
        # title_shape = shapes.placeholders[0]
        title_shape = shapes.title
        title_shape.text = "Overview and agenda"

        #
        body_shape  = shapes.placeholders[1]
        #
        #
        tf = body_shape.text_frame
        # this is a shortcut to adding the first level 0 bullet.
        #tf.text = 'Find the bullet slide layout.'
        #
        # or, for consistency, you canjust do it this way.
        p = tf.add_paragraph()
        p.text = "CPU Activity"
        p.level = 0

        p = tf.add_paragraph()
        p.text = "Active CPUs and jobs"
        p.level=1
        #
        p = tf.add_paragraph()
        p.text = 'Activity breakdown by groups'
        p.level = 1
        #
        p = tf.add_paragraph()
        p.text = 'Job topologies'
        p.level = 1
        #
        p = tf.add_paragraph()
        p.text = 'GPU Activity'
        p.level = 0

        p = tf.add_paragraph()
        p.text = "Active GPUs and jobs"
        p.level=1

        p = tf.add_paragraph()
        p.text = "GPU Activity: By group"
        p.level=1

        p = tf.add_paragraph()
        p.text = "GPU Job Efficienty"
        p.level=1

        if do_save:
            prs.save(report_pathname)
    #
    def slide_active_cpus(self, SACCT_obj=None, SINFO_obj=None, with_trendline=True,
          cpu_jobs=True, gpu_jobs=True, plot_max=True, cpu_col='NCPUS', do_save=False):
        '''
        # computes and plots timeseries of active CPUs. Includes all CPUs, as opposed to CPU-only jobs
        # or CPUs supporting GPU jobs.
        # @with_trendline: include a simple LSQ based trendline over the reporting period.
        '''
        #
        # First, do the computing. We'll do all the slide() things at the end.
        #
        SACCT_obj = SACCT_obj or self.SACCT_obj
        SINFO_obj = SINFO_obj or self.SINFO_obj
        img_path = self.img_path
        #
        if not SINFO_obj is None and plot_max:
            n_cpus_max = 0
            n_gpus_max = 0 
            if cpu_jobs:
                n_cpus_max = SINFO_obj.total_cpu_cpus
            if gpu_jobs:
                n_gpus_max = SINFO_obj.total_gpu_cpus
            n_cpus_total = n_cpus_max + n_gpus_max

        # returns a dataclass object
        # filter report for gpu/cpu with @jobs_summary paramenter:
        js = SACCT_obj.jobs_summary
        # my brain is fried; let's just game out this index, then apply logic.
        #ix_cpu_gpu = numpy.logical_or(numpy.logical_and(ds['NGPUs']>0, gpu_jobs), numpy.logical_and(ds['NGPUs']==0, cpu_jobs))
        ix_gpu = numpy.logical_and(js['NGPUs']>0,  numpy.array([gpu_jobs for _ in js]) )    # these are GPU jobs AND we include GPU jobs
        ix_cpu = numpy.logical_and(js['NGPUs']==0, numpy.array([cpu_jobs for _ in js]))    # these are CPU jobs AND we include CPU jobs
        js = js[numpy.logical_or(ix_gpu, ix_cpu)]
        #
 #       rep_cpu_lc = SACCT_obj.report_activecpus_jobs_layercake_and_CDFs(group_by='Partition',
 #        trendline=with_trendline, cpu_max=(n_cpus_max + n_gpus_max))
        rep_cpu_lc = SACCT_obj.report_activecpus_jobs_layercake_and_CDFs(group_by='Partition',
         trendline=with_trendline, cpu_max=(n_cpus_max + n_gpus_max), jobs_summary=js,
         cpu_col=cpu_col)

        #
        # plot total CPUs in partition. Also, subtract CPUs associated with GPUs. For now, this is just
        #. something we know. we will need to work harder to get it from data.
        fg = rep_cpu_lc.fig
        if cpu_jobs and gpu_jobs:
            jobs_str = "CPU and GPU jobs"
        if cpu_jobs and not gpu_jobs:
            jobs_str = "CPU-only jobs"
        if gpu_jobs and not cpu_jobs:
            jobs_str = "GPU jobs"
        fg.suptitle(f'Active Jobs and CPUs: {jobs_str}', size=16)

        ax = fg.axes[0]
        ln = ax.lines[0]
        x_max_cpus = numpy.array([ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[-1]])
        #
        # max CPUs: Only meaningful if there is some sort of dedicated partition.
        if not SINFO_obj is None and plot_max:
            #n_cpus_max = 0
            #n_gpus_max = 0    # gpu-cpus, not number of GPUs...
            if cpu_jobs:
                #n_cpus_max = SINFO_obj.total_cpu_cpus
                ax.plot( x_max_cpus, numpy.ones(2)*n_cpus_max, ls='--', lw=3.,
                color='m', label='CPUs')
            if gpu_jobs:
                #n_gpus_max = SINFO_obj.total_gpu_cpus
                x_max_cpus = numpy.array([ax.lines[0].get_xdata()[0], ax.lines[0].get_xdata()[-1]])
                ax.plot( x_max_cpus, numpy.ones(2)*(n_cpus_max+n_gpus_max), ls='--', lw=3.,
                    color='r', label='+GPUs') 
        #
        ax.legend(loc='upper left')
        img_filepath = os.path.join(img_path, 'active_cpu_quant.jpg')
        fg.savefig(img_filepath)
        #
        # Now do the slide...
        prs = self.pptx_report
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        #slide = prs.slides.add_slide(prs.slide_layouts[8])

        #slide.shapes.title.text = f"{partition.upper()} Active CPUs and Jobs\n(jobs_str)"
        slide.shapes.title.text = f"{self.cluster_name} Partition: {SACCT_obj.partition} Active CPUs and Jobs: {jobs_str}"
        #
        #
        # figure:
        left = pptx.util.Inches(4)
        top = pptx.util.Inches(2)
        height = pptx.util.Inches(4.75)
        pic = slide.shapes.add_picture(img_filepath, left, top, height=height)
        #
        bullets_text = slide.shapes.placeholders[1]
        bullets_text.left=pptx.util.Inches(.25)
        bullets_text.width=pptx.util.Inches(4)
        bullets_text.top=top
        #
        tf = bullets_text.text_frame
        font_pt=pptx.util.Pt(18)
        #
        p = tf.add_paragraph()
        p.text = f"Quantiles ({(n_cpus_total):,} total CPUs): "
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        #
        # NOTE: We'e asking for errors/exceptions when n_cpus_total is not defined, but maybe
        #. fix it later. Right now, this is in a partition-specific class.
        for k, (q,Q) in enumerate(rep_cpu_lc.Qs_cpus[::-1]):
            p = tf.add_paragraph()
            p.text = f"{int(100*q)}th: {Q:,.2f} ({100*(Q/n_cpus_total):.2f}%)"
            p.alignment = PP_ALIGN.LEFT
            p.level = 1
        #p = tf.add_paragraph()
        #p.text = f"90th pctl, CPUS: {rep_cpu_CPU_jobs.qs_cpus[-1]:,.2f}/{n_cpus_only_serc:,} ({100*(rep_cpu_CPU_jobs.qs_cpus[-1]/n_cpus_only_serc):.2f}%)"
        #p.alignment = PP_ALIGN.LEFT
        #p.level = 0

        if with_trendline:
            p = tf.add_paragraph()
            p.text = f"6 month trend: {'+' if rep_cpu_lc.trend_cpus >=0 else '-'} {rep_cpu_lc.trend_cpus:.2f} cpus/day"
            p.alignment = PP_ALIGN.LEFT
            p.level = 0

        for p in tf.paragraphs[1:]:
            p.runs[0].font.size = font_pt

        if do_save:
            prs.save(report_pathname)

        return rep_cpu_lc
#
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Use this module to run PPTX SACCT reports on Sherlock, or other HPC."
    )

    parser.add_argument('--test', action='store_true', required=False, help='Use this option to test the arg parser and front matter of the module')
    parser.add_argument('--report_type', type=str, required=False, default='partition',
     help='Type of report to run, eg. base, partition, user, group'
     )
     
    parser.add_argument('--report_len_days', type=int, required=False, default=180,
        help='Length of report, in number of days. start_dtm can be calculated from this number'
    )
    
    parser.add_argument(
        '--end_dtm',
        #type=dtm.date,
        type = str,
        required=False,
        default=dtm.datetime.now(),
        help='End date of the reporting period, as a datetime.datetime, in format YYYY-MM-DD'
    )
    
    parser.add_argument(
        '--start_dtm',
        type=str,
        required=False,
        default=None,
        help='Start date of reporting period. Alternatively, report_len_days and end_dtm can be provided and this value is calculated.'
    )
    #
    parser.add_argument("--partition", type=str, required=False, default=None, help="Slurm partition name.")
    parser.add_argument("--group", type=str, required=False, default=None, help="Account/group name.")
    parser.add_argument("--user", dest="user", required=False, type=str, default=None, help="Username to filter on.")
    parser.add_argument("--ncpu", dest="n_cpus", type=int, default=None, help="Number of CPUs the SACCT_obj will use")
    parser.add_argument("--delim_sacct", type=str, default="*", help="Field delimiter for sacct output, usually '*'.")
    parser.add_argument("--sacct-data-path", default="sacct_data", type=str, help="Path to sacct data file.")
    parser.add_argument("--pptx_path", default="pptx_path", type=str, help="Path to store PPTX presentation.")
    parser.add_argument("--pptx_name", default=None, type=str, help="Name of PPTX report. By default, the name will be calculated in code.")
    ####
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    #
    return parser.parse_args(argv)

def info():
    print('** Module to produce partition-type SLURM accounting, PPTX reports.')



def main(argv=None):
    args = parse_args(argv)
    #
    if args.test:
        print('** --test parameter passed.')
        print(f'** partition, group, user, ncpus: {args.partition}, {args.group}, {args.user}, {args.n_cpus}, {args.report_len_days}')
        #
        # must return 0 or the bash script interprets the return as an error.
        return 0
    #
    #for i in range(args.count):
    #    print(f"Hello, {args.name}! ({i + 1}/{args.count})")
    print(f'** partition, group, user, ncpus: {args.partition}, {args.group}, {args.user}, {args.n_cpus}, {args.report_len_days}')
    report_type = args.report_type
    #
    # Put all the variables into a dictionary, so we can mange them more easily? It saves us some coding
    #  for some validation steps, --verbose DEBUG/diagnostic statements, etc. but it adds almost as much
    #  overhead, so let's put a pin in it.
    #
    n_cpus = args.n_cpus
    if n_cpus == 0:
        n_cpus = None
    if not n_cpus is None:
        n_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
    #
    ##########
    # Start and End dates (and runtime length):
    end_dtm         = args.end_dtm
    start_dtm       = args.start_dtm
    report_len_days = args.report_len_days
    # I think these might get automagically converted by argparser.... even though we define them as str?
    if not end_dtm is None and isinstance(end_dtm, str):
        end_dtm = dtm.datetime.strptime(args.end_dtm, "%Y-%m-%d")
    if not start_dtm is None and isinstance(end_dtm, str):
        start_dtm = dtm.datetime.strptime(args.start_dtm, "%Y-%m-%d")
    if not report_len_days is None:
        report_len_days = int(report_len_days)
    #
    # start_dtm = args.start_dtm
    # if isinstance(start_dtm, str):
    #     start_dtm = dtm.datetime.strptime(args.start_dtm, "%Y-%m-%d")
    # #
    # report_len_days = int(args.report_len_days)
    # #
    # # NOTE: We should probably forego this start-end logig here. SACCT_obj should handle this
    # #.  hierarchy, so long as dates are passed as date() and blanks as None
    # # start and end dates: we want a good logic here, so we can provide start + end, start + length, end + length .
    # # start + len
    # if not start_dtm is None and end_dtm is None:
    #     end_dtm = start_dtm + dtm.timedelta(days=report_len_days)
    # # no start, no end:
    # if end_dtm is None and start_dtm is None:
    #     end_dtm = dtm.datetime.now()
    # #
    # # At this point, we have an end_dtm and start_dtm is only None if end_dtm was originally not.
    # if start_dtm is None:
    #     start_dtm = end_dtm - dtm.timedelta(days=report_len_days)
    # #
    # # and for some reason I don't recall, we need date() objects, not datetime():
    # # (but note that in our notebooks, we convert to date(), then subtract tinmedelta, so keep an eye on this...
    # report_len_days = (report_len_days or (end_dtm - start_dtm).days)
    # end_dtm   = end_dtm.date()
    # start_dtm = start_dtm.date()
    ##########
    ##########
    #
    partition   = args.partition
    user        = args.user
    group      = args.group
    delim_sacct = args.delim_sacct
    sacct_data_path = args.sacct_data_path
    pptx_path = (args.pptx_path or 'pptx_path')
    #
    pptx_name = args.pptx_name
    if isinstance(pptx_name, str):
        pptx_name = pptx_name.strip()
    if pptx_name == '':
        pptx_name = None 
    #
    for vr in [partition,user,group,delim_sacct,sacct_data_path,n_cpus]:
        if str(vr).upper() == 'NONE':
            vr = None
            #
    #   #
    #
    verbose = args.verbose
    if verbose:
        print(f"Running with user={user}, partition={partition}, group={group}, start_date={start_dtm}, end_dtm={end_dtm},\
              report_len_days={report_len_days}, delim={delim_sacct}, data_path={sacct_data_path}")
    #
    # TODO: Ok, so put all the PPTX sub classes here, in one .py module? One per report type?
    #. Then, one launcher, with a 'report-flavor' option? 
    #. 1) There is quite a bit to the arg parsing
    #. 2) Looks like there might be more functions abstracted to multiple reports than originally
    #.   anticipated
    #. So let's put it all in one place -- or at least put some in the same place. We'll use
    #. plain-ole if-logic to call the correct report class.
    #
    kwargs_pptx = {
        'partition': partition,
        'group': group,
        'user': user,
        'start_dtm': start_dtm,
        'end_dtm': end_dtm,
        'report_len_days': report_len_days,
        'n_cpus': n_cpus,
        'delim_sacct': delim_sacct,
        'sacct_data_path': sacct_data_path,
        'report_path': pptx_path,
        'report_name': pptx_name,
        'verbose': verbose
        }
    for ky,vl in kwargs_pptx.items():
        #print(f'** **: {ky}, {vl}')
        if str(vl).lower() == "none":
            kwargs_pptx[ky] =  None
        #
    #
#     report = SACCT_PPTX_Deck(partition=partition, group=group, user=user,
#                              start_dtm=start_dtm, end_dtm=end_dtm, report_len_days=report_len_days, n_cpus=n_cpus,
#                           delim_sacct=delim_sacct, sacct_data_path=sacct_data_path,
#                           report_path=pptx_path, report_name=pptx_name
#                             )
# #     , slides_list=None,
# #                           extra_slide_types=None
    if verbose:
        print(f"** DEBUG: Still testing; kwargs:")
        for ky,vl in kwargs_pptx.items():
            print(f'{ky}: {vl}')
    report_type=report_type.lower()
    if report_type == 'base':
        report = SACCT_PPTX_Deck(**kwargs_pptx)
        rep_stat = report.main()
    elif report_type == 'partition':
        print(f'*** Run SACCT_Partition_PPTX_Deck with **kwargs: \n {kwargs_pptx}')
        report = SACCT_Partition_PPTX_Deck(**kwargs_pptx)
        print(f'*** Report instantiated at {report.report_path}. Now, build pages.')
        rep_stat = report.main()
    
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
 
