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

from argparse         import ArgumentParser
from matplotlib.image import imread
from numpy            import zeros, int8, amax, load, savez
from os               import getpid
from os.path          import join
from psutil           import Process
from time             import time
from visualize        import read_descriptions, read_training_expectations

RED         = 0
GREEN       = 1
BLUE        = 2
YELLOW      = 3
NCOLOURS    = 3

colours     = ['red',
               'green',
               'blue',
               'yellow'
              ]

meanings    = ['Microtubules',
               'Protein/antibody',
               'Nuclei channels',
               'Endoplasmic reticulum channels'
              ]

class MemoryReporter:
    def __init__(self):
        self.process = Process(getpid())
        self.initial = self.process.memory_full_info()
        print (f'Process {self.process.pid}, starting with Unique Set Size = {self.initial.rss:,} bytes')
    def check(self):
        full_info = self.process.memory_full_info()
        print (f'{full_info.rss:,} diff= {full_info.rss-self.initial.rss:,}')

if __name__=='__main__':
    start    = time()
    reporter = MemoryReporter()
    parser   = ArgumentParser('Estimate memory usage for down-sampled data, and loading and saving times')
    parser.add_argument('--path',      default = r'd:\data\hpa-scc')
    parser.add_argument('--image_set', default = 'train512x512')

    args         = parser.parse_args()

    Descriptions = read_descriptions('descriptions.csv')
    Training     = read_training_expectations(path=args.path)
    Singletons   = [(image_id, classes) for image_id,classes in Training.items() if len(classes)==1]

    N            = 4096
    mx           = 256
    my           = 256
    Images       = zeros((N,4,mx,my), dtype=int8)
    print (Images.shape, Images.shape[0]*Images.shape[1]*Images.shape[2]*Images.shape[3])
    Targets = []
    for k in range(N):
        if k%32==0:
            print (k)
        image_id,classes = Singletons[k]
        for column,colour in enumerate([BLUE,RED,YELLOW,GREEN]):
            file_name        = f'{image_id}_{colours[colour]}.png'
            path_name        = join(args.path,args.image_set,file_name)
            grey_scale_image = imread(path_name)
            nx,ny            = grey_scale_image.shape
            max_intensity    = amax(grey_scale_image)

            for i in range(mx):
                for j in range(my):
                    if grey_scale_image[2*i,2*j]>0:
                        Images[k,colour,i,j] = int8(128*grey_scale_image[2*i,2*j]/max_intensity)
        Targets.append(classes)

    reporter.check()
    print (Images.shape, Images.shape[0]*Images.shape[1]*Images.shape[2]*Images.shape[3])

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time (building image_set) {minutes} m {seconds:.2f} s')

    start = time()
    savez('test1.npz',Images=Images,Targets=Targets)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time (saving images) {minutes} m {seconds:.2f} s')

    start = time()
    npzfile = load('test1.npz')

    print(npzfile.files)
    print (npzfile['Images'].shape)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time (reloading images){minutes} m {seconds:.2f} s')
