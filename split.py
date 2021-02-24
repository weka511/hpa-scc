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
#  Partition data into training and validation

from argparse import ArgumentParser
from random   import seed,random
from os.path  import splitext,join

if __name__ == '__main__':
    parser = ArgumentParser('Partition data into training and validation')
    parser.add_argument('--path',
                        default = r'd:\data\hpa-scc',
                        help    = 'Path to data')
    parser.add_argument('--file_name',
                        default = 'train.csv',
                        help    = 'List of image ids and classes')
    parser.add_argument('--split',
                        default = 0.9,
                        help    = 'Proportion of data for training')
    parser.add_argument('--seed',
                        default = None,
                        help = 'Seed for random number generator')
    parser.add_argument('--train',
                        default = 'training',
                        help    = 'Training dataset')
    parser.add_argument('--validation',
                        default = 'validation',
                        help    = 'Validation dataset')
    args   = parser.parse_args()

    seed(args.seed)
    training_output_name = args.train if len(splitext(args.train)[1])>0 else f'{args.train}.csv'
    validation_output_name = args.validation if len(splitext(args.validation)[1])>0 else f'{args.validation}.csv'
    i = 0
    j = 0
    k = 0
    with open(join(args.path,args.file_name)) as train,            \
         open(training_output_name,'w')       as training_output,        \
         open( validation_output_name ,'w')   as validation_output:
        for row in train:
            i += 1
            if random()<args.split:
                training_output.write(row)
                j += 1
            else:
                validation_output.write(row)
                k += 1
    print (f'Read {i} records.')
    print (f'Partitioned into {j} training records and {k} for validation')
