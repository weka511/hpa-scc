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
#  To contact me, Simon Crase, email simon@greenweaves.nz
#
#  Find images  that have only one label

from   argparse          import ArgumentParser
from   csv               import writer
from   segment           import read_training_expectations
from   time              import time

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Find images  that have only one label')
    parser.add_argument('--path',
                        default = r'C:\data\hpa-scc',
                        help    = 'Path to data')
    parser.add_argument('--out',
                        default = 'singletons.csv',
                        help    = 'Output file')
    args     = parser.parse_args()
    
    with open(args.out,'w',newline='') as csvfile:
        singleton_writer = writer(csvfile)        
        for image_id,label in sorted([(image_id,labels[0]) 
             for image_id,labels in read_training_expectations(path=args.path).items()
                if len(labels)==1],key=lambda s:s[1]):
            singleton_writer.writerow([label,image_id])
        
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')    