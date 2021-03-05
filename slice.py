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
from math             import ceil
from matplotlib.image import imread
from numpy            import zeros, int8, amax, load, savez
from os.path          import join
from random           import seed, shuffle
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



if __name__=='__main__':
    start    = time()
    parser   = ArgumentParser('Slice and downsample dataset')
    parser.add_argument('output',                                               help='Base name for output datasets')
    parser.add_argument('--path',      default = r'd:\data\hpa-scc',            help='Path where raw data is located')
    parser.add_argument('--image_set', default = 'train512x512',                help='Location of images')
    parser.add_argument('--N',         default = 4096,              type = int, help='Number of images in each output dataset')
    parser.add_argument('--pixels',    default = 256,               type = int, help='Number of pixels')
    parser.add_argument('--seed',                                   type = int, help='Seed for random number generator')
    parser.add_argument('--frequency', default=32,                  type = int, help='Frequency for progress reports')

    args         = parser.parse_args()
    Descriptions = read_descriptions('descriptions.csv')
    Training     = read_training_expectations(path=args.path)
    Singletons   = [(image_id, classes) for image_id,classes in Training.items() if len(classes)==1]
    mx           = args.pixels
    my           = args.pixels
    M            = ceil(len(Singletons)/args.N)
    print (f'Splitting {len(Singletons)} records into {M} files of up to {args.N} slides each, {mx} x {my} pixels')
    seed(args.seed)
    shuffle(Singletons)
    for m in range(M):
        Images       = zeros((args.N,4,mx,my), dtype=int8)

        Targets = []
        for k in range(args.N):
            if k%args.frequency==0:
                print (f'{m}, {k}')
            image_id,classes = Singletons[m*args.N+k]
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

        output = f'{args.output}{m+1}.npz'
        print (f'Saving {output} {Images.shape}')
        savez(output,Images=Images,Targets=Targets)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time (reloading images){minutes} m {seconds:.2f} s')
