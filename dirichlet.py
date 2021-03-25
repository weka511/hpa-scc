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
from os.path           import join
from random            import sample, seed
from scipy.spatial     import Voronoi, voronoi_plot_2d
from sys               import float_info, exc_info
from utils             import Timer

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

def read_descriptions(name):
    with open(name) as descriptions_file:
        return {int(row[0]) : row[1] for row in  reader(descriptions_file)}


# read_training_expectations

def read_training_expectations(path=join(environ['DATA'],'hpa-scc'),file_name='train.csv'):
    with open(join(path,file_name)) as train:
        rows = reader(train)
        next(rows)
        return {row[0] : list(set([int(label) for label in row[1].split('|')])) for row in rows}

def get_dist2(p0,p1):
    x0,y0 = p0
    x1,y1 = p1
    return (x1-x0)**2 + (y1-y0)**2

def get_centroid(Points):
    return (mean([x for x,_ in Points]),
            mean([y for _,y in Points]))

# DPmeans
#
# Find clusters using Dirichlet process means

# Revisiting k-means: New Algorithms via Bayesian Nonparametrics
# Brian Kulis  and Michael I. Jordan
# https://arxiv.org/abs/1111.0352
#
# Parameters:
#     Image
#     Lambda         Penalty
#     background
#     delta
def DPmeans(Image,Lambda=8000,background=0,delta=64):

    def generate_cluster(c):
        return [Xs[i] for i in range(n) if Zs[i]==c]

    def has_converged(mu,mu0):
        return len(mu)==len(mu0) and all(get_dist2(p1,p2)<delta for p1,p2 in zip(sorted(mu),sorted(mu0)))

    # create_observations
    #
    # Generate the Xs. Note that we need to transpose so that scatter (Xs) agree with imshow(...)
    def create_observations():
        return [(i,j) for i in range(nx) for j in range(ny) if Image[j,i]>background]

    nx,ny = Image.shape
    Xs        = create_observations()
    n         = len(Xs)
    k         = 1
    mu        = [get_centroid(Xs)]
    Zs        = [0] * n
    while True:
        for i in range(n):
            D = [get_dist2(Xs[i],mu[c]) for c in range(k)]
            c = argmin(D)
            if D[c] > Lambda:
                Zs[i]     = k
                k         += 1
                mu.append(Xs[i])
            else:
                if Zs[i] != c:
                    Zs[i] = c

        L   =  [generate_cluster(c) for c in range(k)]
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

# get_thinned
#
# Eliminate interior points from component

def get_thinned(Component,n=4):
    # is_isolated
    #
    # Establish whether point is isolated by counting neighbours.

    def is_isolated(x,y):
        count = 0
        for x1,y1 in generate_neighbours(x,y,delta=[-1,0,1]):
            if (x1,y1) in Neighbours:
                count+=1
                if count>n:
                    return False
        return True
    Neighbours = set(Component)
    return [(x,y) for (x,y) in Component if is_isolated(x,y)]

def get_min_distance(C1,C2):
    min_distance = float_info.max
    for pt1 in C1:
        for pt2 in C2:
            min_distance = min(min_distance,get_dist2(pt1,pt2))
    return min_distance

def merge_clusters(k,L,mu,Xs,Zs,delta_max = 64):
    Thinned = [get_thinned(Component) for Component in L]
    Deltas = [(k1,k2,get_min_distance(Thinned[k1],Thinned[k2])) for k1 in range(k) for k2 in range(k1)]

    Pairs = [(k1,k2) for k1,k2,delta in Deltas if delta<delta_max]
    print (Pairs)
    Clusters = {c:[c] for c in range(k)}
    for a,b in Pairs:
        Cluster = Clusters[b]
        Clusters[b].insert(0,a)
        Clusters[a] = Clusters[b]
    Closed = []
    Merged = []
    for a,B in Clusters.items():
        if a in Closed: continue
        Merged.append(B)
        Closed.append(a)
        for b in B:
            Closed.append(b)
    Centroids = []
    for m in Merged:
        Centroids.append(get_centroid([pt for c in list(set(m)) for pt in L[c]]))
    return Thinned, Centroids

def flatten(TT):
    return [t for T in TT for t in T]

def process(image_id  = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0',
            path      = join(environ['DATA'],'hpa-scc'),
            image_set = 'train512x512',
            figs      = '.'):

    Image = Image4(image_id  = image_id,
                   path      = path,
                   image_set = image_set)

    blues = Image.get()
    for seq,(converged,k,L,mu,Xs,Zs) in enumerate(DPmeans(blues[:,:,BLUE])):
        if converged: break

    Thinned, Centroids = merge_clusters(k,L,mu,Xs,Zs)
    voronoi            = Voronoi(Centroids)

    fig   = figure(figsize=(20,20))
    axs   = fig.subplots(nrows = 2, ncols = 2)

    Image.show(axis=axs[0,0])
    axs[0,0].scatter([x for x,_ in mu],[y for _,y in mu],c='m',s=2,alpha=0.5)
    axs[0,0].set_title(f'k={k}, iteration={seq}')

    axs[0,0].scatter([x for x,_ in flatten(Thinned)],[y for _,y in flatten(Thinned)],c='c',s=1,alpha=0.5)

    Image.show(axis=axs[0,1],channels=[RED,BLUE])
    voronoi_plot_2d(voronoi, ax=axs[0,1], show_vertices=False, line_colors='orange')

    Image.show(axis=axs[1,0],channels=[YELLOW,BLUE])
    voronoi_plot_2d(voronoi, ax=axs[1,0], show_vertices=False, line_colors='orange')

    Image.show(axis=axs[1,1],channels=[GREEN,BLUE])
    voronoi_plot_2d(voronoi, ax=axs[1,1], show_vertices=False, line_colors='cyan')

    fig.suptitle(f'{image_id}')
    savefig(join(figs,f'{image_id}_dirichlet'), dpi=args.dpi, bbox_inches='tight')
    return fig

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
    args = parser.parse_args()

    with Timer():
        seed(args.seed)
        Descriptions = read_descriptions('descriptions.csv')
        Training     = restrict(read_training_expectations(path=args.path),
                                labels   = args.labels,
                                multiple = args.multiple)

        if args.sample!=None:
            for image_id in sample(list(Training.keys()),args.sample):
                fig = None
                try:
                    fig = process(image_id  = image_id,
                                  path      = args.path,
                                  image_set = args.image_set,
                                  figs      = args.figs)
                    print (f'Segmented {image_id}')
                except:
                    print(f'Error segmenting {image_id} {exc_info()[0]}')
                finally:
                    if not args.show and fig!=None:
                        close(fig)
        else:
            process(image_id  = args.image_id,
                    path      = args.path,
                    image_set = args.image_set,
                    figs      = args.figs)

    if args.show:
        show()
