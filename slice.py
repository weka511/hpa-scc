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
#  Slice and downsample dataset

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


# create_data
#
# Read training data: optionally filter out some records
#
# Parameters:
#     Training         Training expectations
#     multiplets       Allow data that belongs to multiple classes
#     negative         Allow data that belongs to multiple classes or to no class at all

def create_data(Training,multiplets,negative):
    if negative:
        return [(image_id, classes) for image_id,classes in Training.items()]
    elif multiplets:
        return [(image_id, classes) for image_id,classes in Training.items() if len(classes)>0]
    else:
        return [(image_id, classes) for image_id,classes in Training.items() if len(classes)==1]

# create_image_target
#
# Create one downsampled image & target pair of specified resolution
#
# Parameters:
#     Data      Training data
#     N         Number of images in  output dataset
#     mx        Number of pixels
#     my        Number of pixels
#     start     Identifies first record to be output
#     path      Path where raw data is located
#     imageset  Location of data relative to path

def create_image_target(Data,
                        N         = 1,
                        mx        = 256,
                        my        = 256,
                        start     = 0,
                        path      = r'd:\data\hpa-scc',
                        image_set = 'train512x512'):
    print (f'Creating data: N={N}, start={start}')
    Images  = zeros((N,4,mx,my), dtype=int8)
    Targets = []
    for k in range(N):
        if k%args.frequency==0:
            print (f'{k} of {N} -> {k+start}')
        image_id,classes = Data[k+start]
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
    return Images, Targets

# save_images
#
# Save image/target pair
#
# Parameters:
#     output
#     Images
#     Targets

def save_images(output,Images,Targets):
    print (f'Saving {output} {Images.shape}')
    savez(output,Images=Images,Targets=Targets)



if __name__=='__main__':
    start    = time()
    parser   = ArgumentParser('Slice and downsample dataset')
    parser.add_argument('--output',     default = 'train',                               help = 'Base name for output datasets')
    parser.add_argument('--path',       default = r'd:\data\hpa-scc',                    help = 'Path where raw data is located')
    parser.add_argument('--image_set',  default = 'train512x512',                        help = 'Location of images')
    parser.add_argument('--N',          default = 4096,           type = int,            help = 'Number of images in each output dataset')
    parser.add_argument('--pixels',     default = 256,            type = int,            help = 'Number of pixels after downsampling')
    parser.add_argument('--seed',                                 type = int,            help = 'Seed for random number generator')
    parser.add_argument('--frequency',  default=32,               type = int,            help = 'Frequency for progress reports')
    parser.add_argument('--split',      default = 0.05,           type = float,          help = 'Proportion of data for validation')
    parser.add_argument('--validation', default = 'validation',                          help = 'Validation dataset')
    parser.add_argument('--multiplets', default = False,          action = 'store_true', help = 'Include slides with multiple classes')
    parser.add_argument('--negative', default = False,            action = 'store_true', help = 'Include slides with no classes assigned')
    args         = parser.parse_args()
    Descriptions = read_descriptions('descriptions.csv')
    Data         = create_data(read_training_expectations(path=args.path),
                               args.multiplets,
                               args.negative)
    M            = ceil(len(Data)/args.N)
    print (f'Splitting {len(Data)} records into {M} files of up to {args.N} slides each, {args.pixels} x {args.pixels} pixels')
    seed(args.seed)
    shuffle(Data)
    N_validation = int(args.split*len(Data))
    Images, Targets = create_image_target(Data,
                                          N         =N_validation,
                                          mx        = args.pixels,
                                          my        = args.pixels,
                                          path      = args.path,
                                          image_set = args.image_set)
    save_images(f'{args.validation}.npz',Images,Targets)
    start = N_validation

    for m in range(M):
        N_train = min(len(Data)-start,args.N)
        Images, Targets = create_image_target(Data,
                                              N         = N_train,
                                              mx        = args.pixels,
                                              my        = args.pixels,
                                              start     = start,
                                              path      = args.path,
                                              image_set = args.image_set)
        save_images(f'{args.output}{m+1}.npz',Images,Targets)
        start += N_train

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time (reloading images){minutes} m {seconds:.2f} s')
