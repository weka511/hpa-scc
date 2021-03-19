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
from matplotlib.pyplot import figure, show, savefig, close
from matplotlib.image  import imread
from matplotlib        import cm
from numpy             import zeros, array
from os                import environ
from os.path           import join
from random            import sample, seed


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

# read_descriptions

def read_descriptions(name):
    with open(name) as descriptions_file:
        return {int(row[0]) : row[1] for row in  reader(descriptions_file)}


# read_training_expectations

def read_training_expectations(path=join(environ['DATA'],'hpa-scc'),file_name='train.csv'):
    with open(join(path,file_name)) as train:
        rows = reader(train)
        next(rows)
        return {row[0] : list(set([int(label) for label in row[1].split('|')])) for row in rows}

# visualize
#
# Display images for one slide.

def visualize(image_id     = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0',
              path         = join(environ['DATA'],'hpa-scc'),
              image_set    = 'train512x512',
              figs         = './figs',
              Descriptions = [],
              dpi          = 300):
    fig          = figure(figsize=(20,20))
    axs          = fig.subplots(2, 2)

    for seq,colour in enumerate([BLUE,RED,YELLOW,GREEN]):
        file_name        = f'{image_id}_{colours[colour]}.png'
        path_name        = join(path,image_set,file_name)
        grey_scale_image = imread(path_name)
        nx,ny            = grey_scale_image.shape
        Image            = zeros((nx,ny,NCOLOURS))

        if colour==YELLOW:
            Image[:,:,RED]    = grey_scale_image[:,:]
            Image[:,:,GREEN]  = grey_scale_image[:,:]
        else:
            Image[:,:,colour] = grey_scale_image[:,:]

        axs[seq//2,seq%2].imshow(Image)
        axs[seq//2,seq%2].axes.xaxis.set_ticks([])
        axs[seq//2,seq%2].axes.yaxis.set_ticks([])
        axs[seq//2,seq%2].set_title(meanings[colour])

    fig.suptitle(f'{image_id}: {"+".join([Descriptions[label] for label in Training[image_id]])}')
    savefig(join(figs,image_id), dpi=dpi)
    return fig

# restrict
#
# Used to restrict traing data to specified labels

def restrict(Training,labels):
    return {image_id: image_labels for image_id,image_labels in Training.items() if len(set(image_labels)& set(labels))>0}



if __name__=='__main__':
    parser = ArgumentParser('Visualize HPA data')
    parser.add_argument('--path',
                        default = join(environ['DATA'],'hpa-scc'),
                        help    = 'Folder for data files')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Set resolution of raw images')
    parser.add_argument('--image_id',
                        default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0',
                        help    = 'Used to vew a single image only')
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
                            labels = args.labels)

    if args.all:
        for image_id, cell_types in Training.items():
            if args.multiple or len(cell_types)==1:
                fig = visualize(image_id     = image_id,
                                path         = args.path,
                                image_set    = args.image_set,
                                figs         = args.figs,
                                Descriptions = Descriptions)
                if not args.show:
                    close(fig)
    elif args.sample == None:
        visualize(image_id     = args.image_id,
                  path         = args.path,
                  image_set    = args.image_set,
                  figs         = args.figs,
                  Descriptions = Descriptions)
    else:
        for image_id in sample(list(Training.keys()),args.sample):
            if args.multiple or len(Training[image_id]) == 1:
                fig = visualize(image_id     = image_id,
                                path         = args.path,
                                image_set    = args.image_set,
                                figs         = args.figs,
                                Descriptions = Descriptions)
                if not args.show:
                    close(fig)

    if args.show:
        show()
