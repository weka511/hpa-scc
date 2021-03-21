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
from scipy.signal      import correlate2d,fftconvolve

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
               'Endoplasmic reticulum'
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
              dpi          = 300,
              correlate2d  = False,
              bins         = None):
    fig          = figure(figsize=(20,20))
    nrows        = 2
    if correlate2d:
        nrows += 1
    if bins!=None:
        nrows += 1
    axs          = fig.subplots(nrows = nrows, ncols = NCOLOURS + 1)
    Greys        = []

    for seq,colour in enumerate([BLUE,RED,YELLOW,GREEN]):
        file_name        = f'{image_id}_{colours[colour]}.png'
        path_name        = join(path,image_set,file_name)
        Greys.append(imread(path_name))
        nx,ny            = Greys[-1].shape
        Image            = zeros((nx,ny,NCOLOURS))

        if colour==YELLOW:
            Image[:,:,RED]    = Greys[-1][:,:]
            Image[:,:,GREEN]  = Greys[-1][:,:]
        else:
            Image[:,:,colour] = Greys[-1][:,:]

        axs[0,seq].imshow(Image)
        axs[0,seq].axes.xaxis.set_ticks([])
        axs[0,seq].axes.yaxis.set_ticks([])
        axs[0,seq].set_title(meanings[colour])

    for seq,colour in enumerate([BLUE,RED,YELLOW]):
        nx,ny           = Greys[-1].shape
        Image           = zeros((nx,ny,NCOLOURS))
        Image[:,:,BLUE] = Greys[seq][:,:]
        Image[:,:,RED]  = Greys[GREEN][:,:]
        axs[1,seq].imshow(Image)
        axs[1,seq].axes.xaxis.set_ticks([])
        axs[1,seq].axes.yaxis.set_ticks([])
        axs[1,seq].set_title(f'{meanings[GREEN]}+{meanings[colour]}')
        next_row = 2
        if correlate2d or bins!=None:
            green_reversed = Greys[GREEN][::-1,::-1]
            corr           = fftconvolve(Greys[seq], green_reversed)
            if correlate2d:
                corr_plot = axs[next_row,seq].imshow(corr)
                axs[next_row,seq].axes.xaxis.set_ticks([])
                axs[next_row,seq].axes.yaxis.set_ticks([])
                fig.colorbar(corr_plot, ax=axs[next_row,seq])
                axs[next_row,seq].grid(True)
                next_row+=1
            if bins!=None:
                axs[next_row,seq].hist(corr.flatten(),bins='stone',color=colours[colour])

    for i in range(nrows):
        axs[i,NCOLOURS].axes.xaxis.set_ticks([])
        axs[i,NCOLOURS].axes.yaxis.set_ticks([])

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
    parser.add_argument('--correlate2d',
                        default = False,
                        action  = 'store_true')
    parser.add_argument('--bins')
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
                                correlate2d  = args.correlate2d,
                                bins         = args.bins)
                if not args.show:
                    close(fig)
    elif args.sample == None:
        visualize(image_id     = args.image_id,
                  path         = args.path,
                  image_set    = args.image_set,
                  figs         = args.figs,
                  Descriptions = Descriptions,
                  correlate2d  = args.correlate2d,
                  bins         = args.bins)
    else:
        for image_id in sample(list(Training.keys()),args.sample):
            fig = visualize(image_id     = image_id,
                            path         = args.path,
                            image_set    = args.image_set,
                            figs         = args.figs,
                            Descriptions = Descriptions,
                            correlate2d  = args.correlate2d,
                            bins         = args.bins)
            if not args.show:
                close(fig)

    if args.show:
        show()
