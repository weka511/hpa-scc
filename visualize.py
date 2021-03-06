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

from argparse          import ArgumentParser
from csv               import reader
from hpascc            import *
from matplotlib.pyplot import figure, show, savefig, close
from matplotlib.image  import imread
from matplotlib        import cm
from numpy             import zeros, array
from os                import environ
from os.path           import join
from random            import sample, seed
from sys               import exit

# visualize
#
# Display images for one slide.

def visualize(image_id     = None,
              path         = join(environ['DATA'],'hpa-scc'),
              image_set    = 'train512x512',
              figs         = './figs',
              dpi          = 300,
              Descriptions = {},
              Training     = {}):
    fig          = figure(figsize=(20,20))

    axs          = fig.subplots(nrows = 3, ncols = NCHANNELS)
    Greys        = []

    for seq,colour in enumerate([BLUE,RED,YELLOW,GREEN]):
        file_name        = f'{image_id}_{COLOUR_NAMES[colour]}.png'
        path_name        = join(path,image_set,file_name)
        Greys.append(imread(path_name))
        nx,ny            = Greys[-1].shape
        Image            = zeros((nx,ny,NRGB))

        flattened = [z for zs in Greys[-1] for z in zs if z>0]
        if colour==YELLOW:
            Image[:,:,RED]    = Greys[-1][:,:]/max(flattened)
            Image[:,:,GREEN]  = Greys[-1][:,:]/max(flattened)
        else:
            Image[:,:,colour] = Greys[-1][:,:]/max(flattened)

        axs[0,seq].imshow(Image)
        axs[0,seq].axes.xaxis.set_ticks([])
        axs[0,seq].axes.yaxis.set_ticks([])
        axs[0,seq].set_title(IMAGE_LEVEL_LABELS[colour])

        axs[2,seq].hist(flattened,
                        color = f'xkcd:{COLOUR_NAMES[colour]}',
                        bins  = 25)

    for seq,colour in enumerate([BLUE,RED,YELLOW]):
        nx,ny           = Greys[-1].shape
        Image           = zeros((nx,ny,NRGB))
        Image[:,:,BLUE] = Greys[seq][:,:]
        Image[:,:,RED]  = Greys[GREEN][:,:]
        axs[1,seq].imshow(Image)
        axs[1,seq].axes.xaxis.set_ticks([])
        axs[1,seq].axes.yaxis.set_ticks([])
        axs[1,seq].set_title(f'{IMAGE_LEVEL_LABELS[GREEN]}+{IMAGE_LEVEL_LABELS[colour]}')

    axs[1,NRGB].axes.xaxis.set_ticks([])
    axs[1,NRGB].axes.yaxis.set_ticks([])

    fig.suptitle(f'{image_id}: {"+".join([Descriptions[label] for label in Training[image_id]])}')
    savefig(join(figs,image_id), dpi=dpi, bbox_inches='tight')
    return fig

# restrict
#
# Used to restrict training data to specified labels

def restrict(Training,labels,multiple=False):
    def should_include(image_labels):
        return (len(set(image_labels)& set(labels))>0) and (multiple or len(image_labels)==1)
    return {image_id: image_labels for image_id,image_labels in Training.items() if should_include(image_labels)}



if __name__=='__main__':
    parser = ArgumentParser('Visualize HPA data')
    parser.add_argument('--path',
                        default = join(environ['DATA'],'hpa-scc'),
                        help    = 'Folder for data files')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Set resolution of raw images')
    parser.add_argument('--image_id',
                        default = None,
                        help    = 'Used to view a single image only')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Identifies where to store plots')
    parser.add_argument('--sample',
                        default = None,
                        type    = int,
                        help    = 'Used to sample a specified number of images')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display images')
    parser.add_argument('--multiple',
                        default = False,
                        action  = 'store_true',
                        help    = 'Process images that belong to multiple labels')
    parser.add_argument('--all' ,
                        default = False,
                        action  = 'store_true',
                        help    = 'Process all images, subject to filtering on labels')
    parser.add_argument('--labels',
                        default = list(range(19)),
                        type    = int,
                        nargs   = '*',
                        help    = 'Used to restrict Locations')
    parser.add_argument('--seed',
                        default = None,
                        type    = int,
                        help    = 'Used to seed random number generator')
    parser.add_argument('--dpi',
                        default = 300,
                        type    = int,
                        help    = 'Resolution for saving images')


    args         = parser.parse_args()
    seed(args.seed)
    Descriptions = read_descriptions('descriptions.csv')
    Training     = restrict(read_training_expectations(path=args.path),
                            labels   = args.labels,
                            multiple = args.multiple)

    if args.all:
        for image_id, cell_types in Training.items():
            if args.multiple or len(cell_types)==1:
                fig = visualize(image_id     = image_id,
                                path         = args.path,
                                image_set    = args.image_set,
                                figs         = args.figs,
                                Descriptions = Descriptions,
                                Training     = Training)
                if not args.show:
                    close(fig)
    elif args.sample == None and args.image_id!=None:
        visualize(image_id     = args.image_id,
                  path         = args.path,
                  image_set    = args.image_set,
                  figs         = args.figs,
                  Descriptions = Descriptions,
                  Training     = Training)
    elif args.sample != None:
        for image_id in sample(list(Training.keys()),args.sample):
            fig = visualize(image_id     = image_id,
                            path         = args.path,
                            image_set    = args.image_set,
                            figs         = args.figs,
                            Descriptions = Descriptions,
                            Training     = Training)
            if not args.show:
                close(fig)
    else:
        exit('Missing argument')

    if args.show:
        show()
