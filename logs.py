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

from argparse import   ArgumentParser
from csv               import reader
from matplotlib.pyplot import figure,plot,show,title,legend,ylabel,savefig,ylim,axvspan
from os                import walk
from os.path           import splitext, join

Colours = ['xkcd:red',
           'xkcd:forest green',
           'xkcd:sky blue',
           'xkcd:magenta',
           'xkcd:yellow',
           'xkcd:aqua',
           'xkcd:terracotta'
           ]

def extract(row):
    return row[1]

def is_logfile(filename,prefix='log',suffix='.csv'):
    return filename.startswith(prefix) and splitext(filename)[-1]==suffix

def get_logfile_names(notall,logfiles,prefix='log',suffix='.csv',logdir='logs'):
    if notall:
        return [join(logdir,filename) for filename in logfiles]
    else:
        return [join(logdir,filename) for _,_,filenames in walk(logdir) for filename in filenames if is_logfile(filename,
                                                                                                                prefix=prefix,
                                                                                                                suffix=suffix)]
def set_background(breaks,epochs,facecolor='xkcd:olive'):
    it =iter(breaks)
    try:
        for x in it:
            axvspan(x,next(it),facecolor=facecolor)
    except:
        pass
    for epoch in epochs:
        axvspan(epoch,epoch+5,facecolor='xkcd:black')

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
    parser.add_argument('--suppress',
                        default = False,
                        action  = 'store_true',
                        help    = 'Show averages, but suppress details')
    args     = parser.parse_args()
    fig      = figure(figsize=(20,20))
    i        = 0
    for logfile_name in get_logfile_names(args.notall,args.logfiles,
                                          prefix = args.prefix,
                                          suffix = args.suffix,
                                          logdir = args.logdir):

        with open(logfile_name) as logfile:
            data      = reader(logfile)
            # image_set = extract(next(data))
            lr        = float(extract(next(data)))
            momentum  = float(extract(next(data)))
            losses    = []
            breaks    = []
            epochs    = []
            seq0      = -1
            epoch0    = -1
            for j,[epoch,seq,step,loss] in enumerate(data):
                losses.append(float(loss))
                seq = int(seq)
                if seq != seq0:
                    breaks.append(j)
                    seq0 = seq
                if epoch != epoch0:
                    epochs.append(j)
                    epoch0 = epoch
            breaks.append(j)
            set_background(breaks,epochs)


            if not args.suppress:
                plot (losses[args.skip+args.average:],
                      c     = Colours[i],
                      label = f'lr={lr}, momentum={momentum}')

            plot([sum([losses[i] for i in range(j,j+args.average)])/args.average for j in range(args.skip,len(losses)-args.average)],
                 c     = Colours[i],
                 linestyle = 'dashed',
                 label     = f'lr={lr}, momentum={momentum}' if args.suppress else f'{args.average}-point moving average')

            i += 1
            if i==len(Colours):
                i = 0
    ylabel('Training Error')
    # title(f'Image set: {image_set}')

    legend()
    savefig (f'{args.savefile}.png' if len(splitext(args.savefile))==0 else args.savefile)
    show()
