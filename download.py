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

import kaggle
import os 
import argparse

def download(
    source      = 'train',
    file        = '000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png',
    competition = 'hpa-single-cell-image-classification',
    path        = 'c:\data\hpa-scc',
    colours     = ['green','blue','red','yellow']):
    for colour in colours:
        f = f'{source}/{file}_{colour}.png'
        print(f)
        system(f'kaggle competitions download -f {f}  -c {competition} -p {path}')

def present(file,folder, exts=['png','png.zip']):
    for ext in exts:
        if os.path.isfile(os.path.join(folder,f'{file}.{ext}')):
            return True
    return False
        
if __name__=='__main__':
    parser = argparse.ArgumentParser('Download Kaggle data')
    parser.add_argument('--path',        default = r'c:\data\hpa-scc')
    parser.add_argument('--list',        default = 'train.csv')
    parser.add_argument('--competition', default = 'hpa-single-cell-image-classification')
    parser.add_argument('--source',      default='train')
    args   = parser.parse_args()
    colours     = ['green','blue','red','yellow']
    targets = []
    with open(os.path.join(args.path,args.list)) as files:
        for line in files:
            targets.append(line.split(',')[0])
    full_targets = [f'{target}_{colour}' for target in targets[1:] for colour in colours if not present(f'{target}_{colour}',
                                                                                                        args.path)]
    kaggle.api.authenticate()
    
    for target in full_targets:
        print (f'{args.source}/{target}.png')
        print (f'kaggle competitions download -f {args.source}/{target}  -c {args.competition} -p {args.path}')
        os.system(f'kaggle competitions download -f {args.source}/{target}  -c {args.competition} -p {args.path}')
    #i = 0
    #with open(r'C:\data\hpa-scc\train.csv') as files:
        #for line in files:
            #if i==0: continue
            #download(file=line.split(',')[0])
            #i += 1