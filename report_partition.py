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
import HPC_analytics.hpc_lib as hpc_lib
GB=1024.**3.
#
# end_dtm = dtm.datetime(2026,5,13)
# end_date = end_dtm.date()
# start_date = end_date - dtm.timedelta(days=report_len_days)
# sacct_data_path = 'sacct_data'
# 
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Description of what this script does."
    )
    parser.add_argument(
        '--report_len_days',
        type=int,
        required=False,
        default=180,
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
    parser.add_argument("--partition", type=str, default=None, help="Slurm partition name.")
    parser.add_argument("--group", type=str, default=None, help="Account/group name.")
    parser.add_argument("--user", dest="user", type=str, default=None, help="Username to filter on.")
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

# def run_partition_pptx(partition=None, group=None, user=None, 
#                           start_dtm=None, end_dtm=None, report_len_days=None, n_cpus=None,
#                           delim_sacct=None, sacct_data_path=None,
#                           report_path='pptx_reports', report_name=None
#                       ):
class SACCT_PPXT_Deck(object):
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
                          extra_slide_types=None
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

         SACCT_obj = hpc_lib.get_sacct_direct_or_hdf5(partition=partition, group=group, user=user, 
                               start_dtm=start_dtm, end_dtm=end_dtm, report_len_days=report_len_days, n_cpu=n_cpus,
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
        if report_name is None:
            # TODO: clean up the date strings. Take dates from SACCT_obj?
            report_name = f'{partition}_{str(start_dtm)}_{str(end_dtm)}.ppt'
        report_pathname = os.path.join(report_path, report_name)
        img_path        = os.path.join(report_path, f'images_{report_name}')
        #
        if not os.path.isdir(img_path):
            os.makedirs(img_path)
        #
        if os.path.exists(report_pathname):
            os.remove(report_pathname)
        #
        print(f'** CPUs: {n_cpus_partition}, GPUs: {n_gpus_partition}, cup-CPUs: {n_cpu_cpus_partition}, gpu_CPUs:{n_gpu_cpus_partition}')
        print(f'** Report Path: {report_pathname}')
        SACCT_obj=None
        #
        ####
        #
        # We should now have the data objects we need to compute metrics, make figures, create report.
        #
        # CREATE a Presentation() object:
        prs = pptx.Presentation()
        #
        @dataclasses.dataclass
        class REP:
            sinfo: hpc_lib.SINFO_obj
            sacct: type(SACCT_obj)
        #return REP(SINFO_obj, SACCT_obj)
        REP = REP(SINFO_obj, SACCT_obj)
        #
        self.__dict__.update({ky: vl for ky, vl in locals().items() if ky not in ['self', '__class__'] } )
        #
        #######
        t_slide = self.title_slide()
        a_slide = self.agenda_slide() 
    
    def title_slide(self):
        # TITLE SLIDE:
        prs = self.prs
        #partition = self.partition
        SACCT_obj = self.SACCT_obj
        #
        slide = prs.slides.add_slide(prs.slide_layouts[0])

        slide.shapes.title.text = f"SACCT Reporting for Sherlock Partition: {SACCT_obj.partition}"
        slide.placeholders[1].text = f'({str(SACCT_obj.start_date.date())} - {str(SACCT_obj.end_date.date())})'
        # TODO: If user= or group= or account= are included, not those here? Add a special slide to record
        #. detailed runtime parameters?

        prs.save(self.report_pathname)
        
    def agenda_slide(self):
        prs = self.prs
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

        prs.save(report_pathname)



def main(argv=None):
    args = parse_args(argv)
    
    #for i in range(args.count):
    #    print(f"Hello, {args.name}! ({i + 1}/{args.count})")
    print(f'** {args.partition}, {args.group}, {args.user}, {args.n_cpus}')
        
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
    end_dtm = args.end_dtm
    start_dtm = args.start_dtm
    report_len_days = args.report_len_days
    #
    # start and end dates: we want a good logic here, so we can provide start + end, start + length, end + length .
    # start + len
    if not start_dtm is None and end_dtm is None:
        end_dtm = start_dtm + dtm.timedelta(days=report_len_days)
    # no start, no end:
    if end_dtm is None and start_dtm is None:
        end_dtm = dtm.datetime.now()
    #
    # At this point, we have an end_dtm and start_dtm is only None if end_dtm was originally not.
    if start_dtm is None:
        start_dtm = end_dtm - dtm.timedelta(days=report_len_days)
    #
    # and for some reason I don't recall, we need date() objects, not datetime():
    # (but note that in our notebooks, we convert to date(), then subtract tinmedelta, so keep an eye on this...
    report_len_days = (report_len_days or (end_dtm - start_dtm).days)
    end_dtm   = end_dtm.date()
    start_dtm = start_dtm.date()
    ##########
    #
    partition   = args.partition
    user        = args.user
    group      = args.group
    delim_sacct = args.delim_sacct
    sacct_data_path = args.sacct_data_path
    pptx_path = (args.pptx_path or 'pptx_path')
    pptx_name = args.pptx_name.strip()
    if pptx_name == '':
        pptx_name = None
        
    #
    for vr in [partition,user,group,delim_sacct,sacct_data_path,n_cpus]:
        if str(vr).upper() == 'NONE':
            vr = None
            #
    #   #
    #
    if args.verbose:
        print(f"Running with user={user}, partition={partition}, group={group}, start_date={start_dtm}, end_dtm={end_dtm},\
              report_len_days={report_len_days}, delim={delim_sacct}, data_path={sacct_data_path}")
 
    report = SACCT_PPXT_Deck(partition=partition, group=group, user=user, 
                             start_dtm=start_dtm, end_dtm=end_dtm, report_len_days=report_len_days, n_cpus=n_cpus,
                          delim_sacct=delim_sacct, sacct_data_path=sacct_data_path,
                          report_path=pptx_path, report_name=pptx_name
                            )
#     , slides_list=None,
#                           extra_slide_types=None
    
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
 
