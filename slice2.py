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
#  Slice and segment dataset

from argparse          import ArgumentParser
from dirichlet         import Image4, Mask
from hpascc            import *
from math              import ceil
from matplotlib.image  import imread
from matplotlib.pyplot import figure, close, savefig, show
from numpy             import zeros, int8, amax, load, savez
from os                import environ
from os.path           import join
from random            import seed, shuffle
from utils             import Timer


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
                        path      = join(environ['DATA'],'hpa-scc'),
                        image_set = 'train512x512'):
    print (f'Creating data: N={N}, start={start}')
    Images  = zeros((N,4,mx,my), dtype=int8)
    Targets = []
    for k in range(N):
        if k%args.frequency==0:
            print (f'{k} of {N} -> {k+start}')
        image_id,classes = Data[k+start]
        fig = figure()
        axs = fig.subolots(2,2)
        for column,colour in enumerate([BLUE,RED,YELLOW,GREEN]):
            file_name        = f'{image_id}_{COLOUR_NAMES[colour]}.png'
            path_name        = join(args.path,args.image_set,file_name)
            grey_scale_image = imread(path_name)
            nx,ny            = grey_scale_image.shape
            max_intensity    = amax(grey_scale_image)

            for i in range(mx):
                for j in range(my):
                    if grey_scale_image[2*i,2*j]>0:
                        Images[k,colour,i,j] = int8(128*grey_scale_image[2*i,2*j]/max_intensity)
            axs[column//2][column%2].imshow(Images[k,colour,:,:])
        savefig(fr'\temp\{image_id}-{k}.png')
        close(fig)
        Targets.append(classes)
    return Images, Targets

# save_images
#
# Save image/target pair
#
# Parameters:
#     output      Name of file where data is to be saved
#     Images      Numpy array of images
#     Targets     List of values that are expected

def save_images(output,Images,Targets):
    print (f'Saving {output} {Images.shape}')
    savez(output,Images=Images,Targets=Targets)

class Stats:
    def __init__(self,report_threshold,suspects):
        self.Widths           = []
        self.Heights          = []
        self.Counts           = []
        self.report_threshold = report_threshold
        self.suspects         = suspects
        self.plotfile_name    = 'WidthsHeights.png'

    def __enter__(self):
        self.suspects_file = open(self.suspects,'w')
        return self

    def record_count(self,image_id,count):
        self.Counts.append(count)
        if count<self.report_threshold:
            self.suspects_file.write(f'{image_id},{count}\n')

    def record_rectangle(self,width,height):
        self.Widths.append(width)
        self.Heights.append(height)

    def plot(self):
        fig = figure(figsize=(10,20))
        axs = fig.subplots(nrows=1, ncols=2)
        axs[0].hist([self.Widths, self.Heights],
                    bins  = 25,
                    label = ['Widths', 'Heights'],
                    color = ['red','blue'])
        axs[0].legend(loc='upper right')
        axs[1].hist([self.Counts],
                    bins  = 25,
                    label = ['Counts'],
                    color = ['green'])
        axs[1].legend(loc='upper right')
        savefig(self.plotfile_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.suspects_file.close()


def create_image_target(Data,
                        N            = 1,
                        mx           = 512,
                        my           = 512,
                        path         = join(environ['DATA'],'hpa-scc'),
                        image_set    = 'train512x512',
                        segments     = './segments',
                        Expectations = {},
                        test         = False,
                        stats        = None):
    print (f'Creating data: N={N}')
    Images  = zeros((N,NCHANNELS,mx,my), dtype=float)
    Targets = []
    index   = 0

    for image_id in Batch:
        mask           = Mask.Load(join(args.segments, f'{image_id}.npy'))
        Limits         = mask.get_limits()
        Greys          = [imread(join(path, image_set, f'{image_id}_{COLOUR_NAMES[colour]}.png')) for colour in [BLUE,RED,YELLOW,GREEN]]
        MaxIntensities = [amax(image) for image in Greys]
        stats.record_count(image_id,len(Limits))

        for k in range(len(Limits)):

            i0,j0,i1,j1 = Limits[k]
            stats.record_rectangle(i1-i0,j1-j0)

            for i in range(i0,i1):
                for j in range(j0,j1):
                    if mask[i,j]==k+1:
                        for column in range(len(Greys)):
                            Images[index,column,i-i0,j-j0] = Greys[column][i,j]/MaxIntensities[column]
            print (image_id,k,index)
            if test:
                fig            = figure()
                axs            = fig.subplots(2,2)
                for column in range(len(Greys)):
                    axs[column//2,column%2].imshow(Images[index,column,:,:],
                                                   cmap = 'gray',
                                                   vmin = 0,
                                                   vmax = 1.0)
                savefig(fr'\temp\{image_id}-{k}.png')
                close(fig)
            Targets.append(Expectations[image_id])
            index += 1
    return Images,Targets

if __name__=='__main__':

    parser   = ArgumentParser('Slice and downsample dataset')
    parser.add_argument('--worklist',
                        default = 'worklist',
                        help    = 'Name of file with list of images')
    parser.add_argument('--segments',
                        default = './segments',
                        help    = 'Identifies where cell masks have been stored')
    parser.add_argument('--train',
                        default = 'train',
                        help    = 'Base name for output datasets')
    parser.add_argument('--output',
                        default = './data',
                        help    = 'Identifies where output datasets will be stored')
    parser.add_argument('--path',
                        default = join(environ['DATA'],'hpa-scc'),
                        help    = 'Path where raw data is located')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Location of images')
    parser.add_argument('--N',
                        default = 4096,
                        type    = int,
                        help    = 'Number of images in each output dataset')
    parser.add_argument('--test',
                        default  = False,
                        action   = 'store_true',
                        help     = 'Store plots for verification')
    parser.add_argument('--report_threshold',
                        default = 7,
                        type    = int,
                        help    = 'Report images with fewer segments that this value')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display images')
    args         = parser.parse_args()
    with Timer(), Stats(args.report_threshold,'suspects.csv') as stats:
        Descriptions   = read_descriptions('descriptions.csv')
        Expectations   = read_training_expectations(path=args.path)

        Batch          = []
        total_segments = 0
        file_sequence  = 0
        for image_id in read_worklist(args.worklist):

            mask   = Mask.Load(join(args.segments,f'{image_id}.npy'))
            Limits = mask.get_limits()
            print (image_id,len(Limits))
            if total_segments+len(Limits)>args.N:
                Images,Targets = create_image_target(Batch,
                                                     N            = total_segments,
                                                     segments     = args.segments,
                                                     path         = args.path,
                                                     image_set    = args.image_set,
                                                     Expectations = Expectations,
                                                     test         = args.test,
                                                     stats        = stats)
                file_sequence += 1
                save_images(join(args.output,f'{args.train}{file_sequence}.npz'),Images,Targets)
                Batch.clear()
                total_segments = 0
            total_segments += len(Limits)
            Batch.append(image_id)
        if len(Batch)>0:
            Images,Targets= create_image_target(Batch,
                                                N            = total_segments,
                                                segments     = args.segments,
                                                path         = args.path,
                                                image_set    = args.image_set,
                                                Expectations = Expectations,
                                                test         = args.test,
                                                stats        = stats)
            file_sequence += 1
            save_images(join(args.output,f'{args.train}{file_sequence}.npz'),Images,Targets)

        stats.plot()

    if args.show:
        show()