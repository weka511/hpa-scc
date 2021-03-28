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
from matplotlib.pyplot import hist, show, figure, savefig, close
from matplotlib.image  import imread
from numpy             import zeros, mean, std, argmin
from os                import environ
from os.path           import exists,join
from random            import sample
from re                import split
from scipy.spatial     import Voronoi, voronoi_plot_2d
from sys               import float_info, exc_info, exit
from utils             import Logger, set_random_seed, Timer

RED                = 0      # Channel number for Microtubules
GREEN              = 1      # Channel number for Protein of interest
BLUE               = 2      # Channel number for Nucelus
YELLOW             = 3      # Channel number for Endoplasmic reticulum
NCHANNELS          = 4      # Number of channels
NRGB               = 3      # Number of actual channels for graphcs

COLOUR_NAMES     = [
    'red',
    'green',
    'blue',
    'yellow'
]

IMAGE_LEVEL_LABELS    = [
    'Microtubules',
    'Protein/antibody',
    'Nuclei channels',
    'Endoplasmic reticulum'
]

class Image4(object):
    def __init__(self,
                 path        = join(environ['DATA'],'hpa-scc'),
                 image_set   = 'train512x512',
                 image_id    = '000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0'):

        for channel in range(NCHANNELS):
            image_mono = imread(join(path,
                                     image_set,
                                     f'{image_id}_{COLOUR_NAMES[channel]}.png'))
            if channel==0:
                self.nx,self.ny     = image_mono.shape
                self.Image          = zeros((self.nx,self.ny,NCHANNELS))

            self.Image[:,:,channel] = image_mono

    def get(self,channels=[BLUE]):
        Image            = zeros((self.nx,self.ny,NRGB))
        for channel in channels:
            if channel==YELLOW:
                Image[:,:,RED]    =  self.Image [:,:,channel]
                Image[:,:,GREEN]  =  self.Image [:,:,channel]
            else:
                Image[:,:,channel] =  self.Image [:,:,channel]
        return Image

    def show(self,
            channels = [BLUE],
            axis     = None,
            origin   = 'lower'):
        axis.imshow(self.get(channels),
                    extent = [0,self.nx-1,0,self.ny-1],
                    origin = origin)
        axis.set_xlim(0,self.nx-1)
        axis.set_ylim(0,self.ny-1)


# restrict
#
# Used to restrict training data to specified labels

def restrict(Training,labels,multiple=False):
    def should_include(image_labels):
        return (len(set(image_labels)& set(labels))>0) and (multiple or len(image_labels)==1)
    return {image_id: image_labels for image_id,image_labels in Training.items() if should_include(image_labels)}

# read_descriptions
#
# Create a lookup table for the inerpretation of each label

def read_descriptions(file_name='descriptions.csv'):
    with open(file_name) as descriptions_file:
        return {int(row[0]) : row[1] for row in  reader(descriptions_file)}


# read_training_expectations

def read_training_expectations(path=join(environ['DATA'],'hpa-scc'),file_name='train.csv'):
    with open(join(path,file_name)) as train:
        rows = reader(train)
        next(rows)
        return {row[0] : list(set([int(label) for label in row[1].split('|')])) for row in rows}

# get_dist_sq
#
# Compute squared Euclidean distance between two points

def get_dist_sq(pt0,pt1):
    x0,y0 = pt0
    x1,y1 = pt1
    return (x1-x0)**2 + (y1-y0)**2

# get_centroid
#
# Determine centroids of a list of points

def get_centroid(Points):
    return (mean([x for x,_ in Points]),
            mean([y for _,y in Points]))

# DPmeans
#
# Find clusters using Dirichlet process means
#
# Revisiting k-means: New Algorithms via Bayesian Nonparametrics
# Brian Kulis  and Michael I. Jordan
# https://arxiv.org/abs/1111.0352
#
# Parameters:
#     Image        Image to be segmented
#     Lambda       Penalty to discourage creating new clusters
#     background   Threshold for blue pixels. We consider pixels only if they exceed this value.
#     delta        Used to assess convergence. If no centroids have shifted by more than this limit, we have converged

def DPmeans(Image,
            Lambda     = 8000,
            background = 0,
            delta      = 64):

    # extract_one_cluster
    #
    # Extract observations that should be assigned to specified cluster index
    def extract_one_cluster(index):
        return [Xs[i] for i in range(n) if Zs[i]==index]

    # has_converged
    #
    # Establish whether iterations have converged, i.e.: number of clusters hasn't changed since last iteration
    # and the centroids haven't moved by a distance of mre than sqrt(delta)

    def has_converged(mu,mu0):
        return len(mu)==len(mu0) and all(get_dist_sq(p1,p2)<delta for p1,p2 in zip(sorted(mu),sorted(mu0)))

    # create_observations
    #
    # Generate the Xs. Note that we need to transpose so that scatter(Xs...) agree with imshow(...)

    def create_observations():
        return [(i,j) for i in range(nx) for j in range(ny) if Image[j,i]>background]

    nx,ny = Image.shape
    Xs    = create_observations()
    n     = len(Xs)
    k     = 1                    # Initially all points are in just one cluster
    mu    = [get_centroid(Xs)]   # So there is only one centroid
    Zs    = [0] * n              # Hidden variables - initially all points are in just one cluster

    while True:
        for i in range(n):
            D = [get_dist_sq(Xs[i],mu[c]) for c in range(k)]
            c = argmin(D)
            if D[c] > Lambda:     # Create new cluster
                Zs[i]     = k
                k         += 1
                mu.append(Xs[i])
            else:                  # Assign point to closest cluster
                if Zs[i] != c:
                    Zs[i] = c

        L   =  [extract_one_cluster(c) for c in range(k)]
        mu0 = mu[:]
        mu  = [get_centroid(Xs) for Xs in L]
        yield has_converged(mu,mu0),k,L,mu,Xs,Zs

# generate_neighbours
#
# Used to iterate through 8 neighbours of a point
#
# Parameters:
#     x         x coordinate of point
#     y         y coordinate of point
#     delta     Used to generate neighbours from cartesian product delta x delta

def generate_neighbours(x,y,delta=[-1,0,1]):
    for dx in delta:
        for dy in delta:
            if dx==0 and dy==0: continue  # Don't visit the original point
            yield x + dx, y + dy


# remove_isolated_centroids
#
# Remove centroids that have fewer than a specified number of points

def remove_isolated_centroids(L,mu,cutoff = 20):
    mu = [m for m,cluster in zip(mu,L) if len(cluster) > cutoff]
    L  = [cluster for cluster in L if len(cluster) > cutoff]
    return len(L),mu,L


# create_xkcd_colours
#
# Create list of XKCD colours
def create_xkcd_colours(file_name='rgb.txt'):
    with open(file_name) as colours:
        for row in colours:
            parts = split(r'\s+#',row)
            if len(parts)>1:
                yield parts[0]

def get_image_file_name(image_id, figs = '.'):
    return join(figs,f'{image_id}_dirichlet.png')

# segment
#
#  Read slides and segment image: start with Dirichlet process means, then postprocess to remove phantom cells.
#
#     image_id     The slide to be procesed
#     path         Location of images
#     image_set    Set resolution of raw images
#     figs         Where to store figures
#     Descriptions Map numerical labels into description
#     Training     Map image name to expected labels
#     Lambda       Penalty to discourage creating new clusters
#     background   Threshold for blue pixels. We consider pixels only if they exceed this value.
#     delta        Used to assess convergence. If no centroids have shifted by more than this limit, we have converged

def segment(image_id     = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0',
            path         = join(environ['DATA'],'hpa-scc'),
            image_set    = 'train512x512',
            figs         = '.',
            Descriptions = [],
            Training     = [],
            Lambda       = 8000,
            background   = 0,
            delta        = 64,
            XKCD         = [],
            N            = 100):

    Image = Image4(image_id  = image_id,
                   path      = path,
                   image_set = image_set)
    seq = None
    for seq,(converged,k,L,mu,Xs,Zs) in enumerate(DPmeans(Image.get()[:,:,BLUE],
                                                          Lambda     = Lambda,
                                                          background = background,
                                                          delta      = delta)):
        if converged: break

        if seq>N:
            print (f'Failed to converge in {N} steps')
            break

    k,mu,L             = remove_isolated_centroids(L,mu)

    voronoi            = Voronoi(mu)

    fig                = figure(figsize=(20,20))
    axs                = fig.subplots(nrows = 2, ncols = 2)

    Image.show(axis=axs[0,0])
    for l in range(len(L)):
        axs[0,0].scatter([x for x,_ in L[l]],[y for _,y in L[l]],c=f'xkcd:{XKCD[l]}',s=1)
    axs[0,0].scatter([x for x,_ in mu],[y for _,y in mu],marker='X',c=f'xkcd:{XKCD[k+1]}',s=10)
    axs[0,0].set_title(f'k={k}, iteration={seq}')

    Image.show(axis=axs[0,1],channels=[RED,BLUE])
    voronoi_plot_2d(voronoi, ax=axs[0,1], show_vertices=False, line_colors='orange')

    Image.show(axis=axs[1,0],channels=[YELLOW,BLUE])
    voronoi_plot_2d(voronoi, ax=axs[1,0], show_vertices=False, line_colors='orange')

    Image.show(axis=axs[1,1],channels=[GREEN,BLUE])
    voronoi_plot_2d(voronoi, ax=axs[1,1], show_vertices=False, line_colors='cyan')

    fig.suptitle(f'{image_id}: {", ".join([Descriptions[label] for label in Training[image_id]])}')
    savefig(get_image_file_name(image_id,figs=figs),
            dpi         = args.dpi,
            bbox_inches = 'tight')
    return fig,seq



if __name__=='__main__':
    parser = ArgumentParser('Cluster HPA data using Dirichlet Process Means')
    parser.add_argument('--path',
                        default = join(environ['DATA'],'hpa-scc'),
                        help    = 'Folder for data files')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Set resolution of raw images')
    parser.add_argument('--image_id',
                        default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0',
                        help    = 'Used to view a single image only')
    parser.add_argument('--figs',
                        default = './figs',
                        help    = 'Identifies where to store plots')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display images')
    parser.add_argument('--dpi',
                        default = 300,
                        type    = int,
                        help    = 'Resolution for saving images')
    parser.add_argument('--min_count',
                        default = 10,
                        type    = int,
                        help    = 'Discard clusters with fewer points')
    parser.add_argument('--sample',
                        default = None,
                        type    = int,
                        help    = 'Used to sample a specified number of images')
    parser.add_argument('--multiple',
                        default = False,
                        action  = 'store_true',
                        help    = 'Process images that belong to multiple labels')
    parser.add_argument('--labels',
                        default = list(range(19)),
                        type    = int,
                        nargs   = '*',
                        help    = 'Used to restrict Locations')
    parser.add_argument('--seed',
                        default = None,
                        type    = int,
                        help    = 'Used to seed random number generator')
    parser.add_argument('--Lambda',
                        type    = float,
                        default = 8000,
                        help    = 'Penalty to discourage creating new clusters')
    parser.add_argument('--background',
                        type    = float,
                        default = 0.05,
                        help    = 'Threshold for blue pixels. We consider pixels only if they exceed this value.')
    parser.add_argument('--delta',
                        type    = float,
                        default = 1,
                        help    = 'If no centroids have shifted by more than this limit, we deem iterations to have converged')
    parser.add_argument('--all',
                        default = False,
                        action = 'store_true',
                        help   = 'Used to process all unprocessed slides')
    parser.add_argument('--N',
                        default = 100,
                        type    = int,
                        help    = 'Maximum number of iterations for DPmeans')
    args = parser.parse_args()

    with Timer(),Logger('dirichlet') as logger:
        XKCD         = [colour for colour in create_xkcd_colours()][::-1]
        Descriptions = read_descriptions('descriptions.csv')
        Training     = restrict(read_training_expectations(path=args.path),
                                labels   = args.labels,
                                multiple = args.multiple or args.sample==None)

        if args.all:
            for image_id in Training.keys():
                if exists(get_image_file_name(image_id,figs=args.figs)): continue
                fig = None
                try:
                    fig,seq = segment(image_id     = image_id,
                                      path         = args.path,
                                      image_set    = args.image_set,
                                      figs         = args.figs,
                                      Descriptions = Descriptions,
                                      Training     = Training,
                                      background   = args.background,
                                      XKCD         = XKCD,
                                      N            = args.N,
                                      delta        = args.delta)
                    print (f'Segmented {image_id},{seq}')
                    logger.log(f'{image_id},{seq}')
                except KeyboardInterrupt:
                    exit(f'Interrupted while segmenting {image_id}')
                except:
                    print(f'Error segmenting {image_id} {exc_info()[0]}')
                finally:
                    if not args.show and fig!=None:
                        close(fig)
        elif args.sample!=None:
            set_random_seed(args.seed)
            for image_id in sample(list(Training.keys()),args.sample):
                fig = None
                try:
                    fig,seq = segment(image_id     = image_id,
                                  path         = args.path,
                                  image_set    = args.image_set,
                                  figs         = args.figs,
                                  Descriptions = Descriptions,
                                  Training     = Training,
                                  background   = args.background,
                                  XKCD         = XKCD,
                                  N            = args.N,
                                  delta        = args.delta)
                    print (f'Segmented {image_id},{seq}')
                    logger.log(f'{image_id},{seq}')
                except KeyboardInterrupt:
                    exit(f'Interrupted while segmenting {image_id}')
                except:
                    print(f'Error segmenting {image_id} {exc_info()[0]}')
                finally:
                    if not args.show and fig!=None:
                        close(fig)
        else:
            _,seq =segment(image_id     = args.image_id,
                           path         = args.path,
                           image_set    = args.image_set,
                           figs         = args.figs,
                           Descriptions = Descriptions,
                           Training     = Training,
                           background   = args.background,
                           XKCD         = XKCD,
                           N            = args.N,
                           delta        = args.delta)
            print (f'Segmented {args.image_id},{seq}')
            logger.log(f'{args.image_id},{seq}')

    if args.show:
        show()
