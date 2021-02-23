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
from matplotlib.pyplot import figure,plot,show,title,legend,ylabel,savefig
from os.path           import splitext

def extract(row):
    return row[1]

if __name__ == '__main__':
    parser = ArgumentParser('Analyze log files from training/testing')
    parser.add_argument('--logfiles', nargs='+', default = ['log.csv'])
    parser.add_argument('--savefile', default='logs')
    args = parser.parse_args()

    fig  = figure(figsize=(20,20))
    for logfile_name in args.logfiles:
        with open(logfile_name) as logfile:
            data      = reader(logfile)
            image_set = extract(next(data))
            lr        = float(extract(next(data)))
            momentum  = float(extract(next(data)))
            errors    = [float(error) for _,_,error in data]
            plot (errors,label=f'lr={lr}, momentum={momentum}')

    ylabel('Training Error')
    title(f'Image set: {image_set}')
    legend()
    savefig (f'{args.savefile}.png' if len(splitext(args.savefile))==0 else args.savefile)
    show()
