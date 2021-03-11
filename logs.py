#   Copyright (C) 2021 Greenweaves Software Limited
#
#   This program is free software: you can redistribute it and/or
#   modify it under the terms of the GNU Lesser General Public
#   License as published by the Free Software Foundation; either
#   version 2.1 of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU  Lesser General Public License for more details
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Analyze log files from training/testing

from argparse          import   ArgumentParser
from csv               import reader
from matplotlib.pyplot import figure,plot,show,title,legend,ylabel,savefig,ylim,axvspan
from os                import walk
from os.path           import splitext, join


# is_logfile
#
# Verify whether file name is a valid logfile

def is_logfile(filename,prefix='log',suffix='.csv'):
    return filename.startswith(prefix) and splitext(filename)[-1]==suffix

# get_logfile_names
#
# Used to determine which files are to be processed

def get_logfile_names(notall,logfiles,prefix='log',suffix='.csv',logdir='logs'):
    return                                                                     \
           [join(logdir,filename) for filename in logfiles] if notall  else    \
           [join(logdir,filename) for _,_,filenames in walk(logdir)            \
            for filename in filenames if is_logfile(filename,
                                                    prefix=prefix,
                                                    suffix=suffix)]
# set_background
#
# show epoch and file boundaries
def set_background(breaks,epochs,facecolor='xkcd:olive'):
    it =iter(breaks)
    try:
        for x in it:
            axvspan(x,next(it),facecolor=facecolor)
    except:
        pass
    for epoch in epochs:
        axvspan(epoch,epoch+5,facecolor='xkcd:black')

# create_parameters
#
# Extract parametrs from log file (variable number)

def create_parameters(data):
    product    = []

    while True:
        row = next(data)
        key = row[0]
        if key.isnumeric(): break
        product.append((key,row[1]))

    return product

# display_parameters
#
# Display paramters from logfile - used in legend

def display_parameters(params):
    return ', '.join([f'{key}={value}' for key,value in params])

if __name__ == '__main__':
    parser = ArgumentParser('Analyze log files from training/testing')
    parser.add_argument('--logfiles',
                        default = ['log.csv'],
                        nargs   = '+',
                        help    = 'List of files to be plotted (if --notall)')
    parser.add_argument('--prefix',
                        default = 'log',
                        help    = 'Specifes pattern for logfiles (with suffix)')
    parser.add_argument('--suffix',
                        default = '.csv',
                        help    = 'Specifes pattern for logfiles (with prefix)')
    parser.add_argument('--notall',
                        default = False,
                        action  = 'store_true',
                        help    = 'Controls whther we plot all log files, or merely those specifed by --logfiles')
    parser.add_argument('--savefile',
                        default = 'logs',
                        help    = 'File name for saving plot')
    parser.add_argument('--logdir',
                        default = './logs',
                        help    = 'directory for storing logfiles')
    parser.add_argument('--skip',
                        default = 10,
                        type    = int,
                        help    = 'Number of burn-in entries to be skipped at beginning')
    parser.add_argument('--average',
                        default = 10,
                        type    = int,
                        help    = 'Plot moving average')
    parser.add_argument('--detail',
                        default = False,
                        action  = 'store_true',
                        help    = 'Show details, not just averages')
    parser.add_argument('--colours',
                        default  = ['xkcd:red',
                                    'xkcd:forest green',
                                    'xkcd:sky blue',
                                    'xkcd:magenta',
                                    'xkcd:yellow',
                                    'xkcd:aqua',
                                    'xkcd:terracotta'
                                    ],
                        help     = 'Colours for plots')
    args     = parser.parse_args()
    fig      = figure(figsize=(20,20))
    i        = 0
    for logfile_name in get_logfile_names(args.notall,args.logfiles,
                                          prefix = args.prefix,
                                          suffix = args.suffix,
                                          logdir = args.logdir):

        with open(logfile_name) as logfile:
            data      = reader(logfile)
            params    = create_parameters(data)
            losses    = []
            breaks    = []
            epochs    = []
            seq0      = -1
            epoch0    = -1
            try:
                for j,[epoch,seq,step,loss] in enumerate(data):
                    losses.append(float(loss))
                    seq = int(seq)
                    if seq != seq0:
                        breaks.append(j)
                        seq0 = seq
                    if epoch != epoch0:
                        epochs.append(j)
                        epoch0 = epoch
            except ValueError:  # This can happen if training interrupted
                pass
            finally:
                breaks.append(j)
                set_background(breaks,epochs)

            if args.detail:
                plot (losses[args.skip+args.average:],
                      c     = args.colours[i],
                      label = display_parameters(params))

            plot([sum([losses[i] for i in range(j,j+args.average)])/args.average for j in range(args.skip,len(losses)-args.average)],
                 c         = Colours[i],
                 linestyle = 'dashed',
                 label     = display_parameters(params) if not args.detail else f'{args.average}-point moving average')

            i += 1
            if i==len(Colours):
                i = 0
    ylabel('Training Error')

    legend()
    savefig (f'{args.savefile}.png' if len(splitext(args.savefile))==0 else args.savefile)
    show()
