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
#  1. Find images  that have only one label
#  2. Prepare list of files for segment.py, making sure that specified types/classes are present

from   argparse          import ArgumentParser
from   csv               import writer
from   random            import shuffle
from   segment           import read_training_expectations,create_descriptions
from   time              import time

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Find images  that have only one label')
    parser.add_argument('--path',
                        default = r'd:\data\hpa-scc',
                        help    = 'Path to data')
    parser.add_argument('--out',
                        default = 'singletons.csv',
                        help    = 'Output file')
    parser.add_argument('--selection',
                        default = 'selection.txt',
                        help    = 'Selection of files to be processed by segment.py')
    parser.add_argument('--classes',
                        default = ['all'],
                        nargs   = '+',
                        help    = 'List of classes to be processed by segment.py')
    parser.add_argument('--n',
                        default = 1,
                        type    = int,
                        help    = 'Number of instances of each class to be processed by segment.py')
    args     = parser.parse_args()
    descriptions = create_descriptions()
    with open(args.out,'w',newline='') as csvfile:
        singleton_writer = writer(csvfile)
        for image_id,label in sorted([(image_id,labels[0])
             for image_id,labels in read_training_expectations(path=args.path).items()
                if len(labels)==1],key=lambda s:s[1]):
            singleton_writer.writerow([label,image_id])

    if len(args.classes)>0:
        classes = list(range(len(descriptions)+1)) if args.classes[0]=='all' else args.classes
        counts  = [0 for _ in classes]
        with open(args.selection,'w') as out:
            singletons   = {image_id:label[0] for image_id,label in read_training_expectations(path=args.path).items() if len(label)==1}
            file_names   = list(singletons.keys())
            shuffle(file_names)
            for i in range(len(file_names)):
                if min(counts)>=args.n: break
                j = singletons[file_names[i]]
                if j < len(counts) and counts[j]<args.n:
                    out.write(f'{file_names[i]}\n')
                    counts[j] += 1

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
