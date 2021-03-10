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
from numpy             import zeros, array, mean, log
from numpy.fft         import fft2
from os                import environ
from os.path           import join
from random            import sample


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
              Descriptions = []):
    fig          = figure(figsize=(20,20))
    axs          = fig.subplots(2, 4)

    for column,colour in enumerate([BLUE,RED,YELLOW,GREEN]):
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

        axs[0,column].imshow(Image)
        axs[0,column].axes.xaxis.set_ticks([])
        axs[0,column].axes.yaxis.set_ticks([])
        axs[0,column].set_title(meanings[colour])

        freqs = fft2(grey_scale_image)
        axs[1,column].imshow(log(abs(freqs)),cmap='gray')

    fig.suptitle(f'{image_id}: {"+".join([Descriptions[label] for label in Training[image_id]])}')
    savefig(join(figs,image_id))
    return fig

if __name__=='__main__':
    parser = ArgumentParser('Visualize HPA data')
    parser.add_argument('--path',      default = join(environ['DATA'],'hpa-scc'))
    parser.add_argument('--image_set', default = 'train512x512')
    parser.add_argument('--image_id',  default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0')
    parser.add_argument('--figs',      default = './figs',  help    = 'Identifies where to store plots')
    parser.add_argument('--sample',    default = None, type=int)
    parser.add_argument('--show',      default = False, action='store_true')
    parser.add_argument('--multiple',  default = False, action='store_true')
    parser.add_argument('--all' ,      default = False, action='store_true')
    args         = parser.parse_args()

    Descriptions = read_descriptions('descriptions.csv')
    Training     = read_training_expectations(path=args.path)

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
