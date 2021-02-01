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

from   os.path          import join,basename
from   argparse         import ArgumentParser
from   time             import time

Descriptions = [
    'Nucleoplasm',
    'Nuclear membrane',
    'Nucleoli',
    'Nucleoli fibrillar center',
    'Nuclear speckles',
    'Nuclear bodies',
    'Endoplasmic reticulum',
    'Golgi apparatus',
    'Intermediate filaments',
    'Actin filaments',
    'Microtubules',
    'Mitotic spindle',
    'Centrosome',
    'Plasma membrane',
    'Mitochondria',
    'Aggresome',
    'Cytosol',
    'Vesicles and punctate cytosolic patterns',
    'Negative'
]



def read_training_expectations(path=r'C:\data\hpa-scc',file_name='train.csv'):
    header    = True
    Types     = [[] for _ in Descriptions]
    for line in open(join(path,file_name)):
        if header:
            header = False
            continue
        trimmed = line.strip().split(',')
        for l in trimmed[1].split('|'):
            Types[int(l)].append(trimmed[0])
 
    return Types

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('List datasets for each type')
    parser.add_argument('--path',  default=r'C:\data\hpa-scc')
    args     = parser.parse_args()
    
    Types = read_training_expectations(path=args.path)
    with open ('summary.txt','w') as summary:
        for index in range(len(Types)):
            summary.write(f'{index} {Descriptions[index]}\n')
            for file in Types[index]:
                summary.write (f'\t{file}\n')
 
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
