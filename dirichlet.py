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

from argparse           import ArgumentParser
from hpascc             import *
from math               import sqrt
from matplotlib.colors  import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.pyplot  import hist, show, figure, savefig, close
from matplotlib.image   import imread
from numpy              import zeros, mean, std, argmin, save, load
from os                 import environ
from os.path            import exists,join
from random             import shuffle
from re                 import split
from scipy.spatial      import Voronoi, voronoi_plot_2d
from sys                import float_info, exc_info, exit
from utils              import Logger, Timer, create_xkcd_colours, set_random_seed
from warnings           import filterwarnings



# Image4
#
# This class manages a complete set of images for one slide

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

    # get
    #
    # Get a regular 3 colour image from selected channels,
    # which includes merging YELLOW into RED and GREEN using weights from
    # https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/214863

    def get(self,channels=[BLUE]):
        Image = zeros((self.nx,self.ny,NRGB))
        for channel in channels:
            if channel==YELLOW:
                Image[:,:,RED]    +=  self.Image [:,:,channel]
                Image[:,:,GREEN]  +=  0.5* self.Image [:,:,channel]
            else:
                Image[:,:,channel] +=  self.Image [:,:,channel]
        return Image

    # get_segment
    #
    # Factory method, currrent used only by spikes, to prepare data for display

    def get_segment(self,
                    channels = [BLUE],
                    Mask     = [],
                    selector = None):
        Product = zeros((self.nx,self.ny,NRGB))
        for i in range(self.nx):
            for j in range(self.ny):
                if Mask[i,j]==selector:
                    for channel in channels:
                        if channel==YELLOW:
                            Product[i,j,RED]     =  self.Image [i,j,channel]
                            Product[i,j,GREEN]   =  self.Image [i,j,channel]
                        else:
                            Product[i,j,channel] =  self.Image [i,j,channel]
        return Product

    # show
    #
    #  Display selected channels
    #
    #  Parameters:
    #      channels      Channels to be displayed
    #      axis          Identifies where plot is to be displayed
    #      actuals       Controls colours used for display, e.g. map nucleus to BLUE, some other channel to RED
    #

    def show(self,
            channels = [BLUE],
            axis     = None,
            actuals  = None):

        if actuals==None:
            axis.imshow(self.get(channels),
                        extent = [0,self.nx-1,0,self.ny-1],
                        origin = 'lower')
        else:
            Image = zeros((self.nx,self.ny,NRGB))
            for a,b in zip(channels,actuals):
                Image[:,:,b] = self.Image[:,:,a]
            axis.imshow(Image,
                        extent = [0,self.nx-1,0,self.ny-1],
                        origin = 'lower')
        axis.set_xlim(0,self.nx-1)
        axis.set_ylim(0,self.ny-1)


# SegmentationMask
#
# This class represents a segmentation mask

class SegmentationMask:

    # Load
    #
    # A factory method for [re]creating a Mask that has been saved previously
    #
    # Parameters:
    #     file_name    File where mask has been stored

    @classmethod
    def Load(cls,file_name,dist=None):
        Product               = SegmentationMask(dist=dist)
        Product.Mask          = load(file_name)
        Product.nx,Product.ny = Product.Mask.shape
        return Product

    def __init__(self,Image=None,Centroids=[],dist=None):
        if Image is None: return            # Used by Mask.Load(...) to create empty Mask
        self.nx    = Image.nx
        self.ny    = Image.ny
        if len(Centroids)>0:
            self.Mask  = self.create_bit_mask(Image,Centroids,dist=dist)

    # create_bit_mask
    #
    # Create a mask that has the same size as image, and initialize each
    # point with the index (1-based) of the nearest centroid

    def create_bit_mask(self,Image,Centroids,dist=None):
        Product = zeros((Image.ny,Image.ny))
        for i in range(Image.nx):
            for j in range(Image.ny):
                max_distance = float_info.max
                for k in range(1,1+len(Centroids)):
                    distance = dist((i,j),Centroids[k-1])
                    if distance<max_distance:
                        Product[i,j] = k
                        max_distance = distance
        return Product

    # __getitem__
    #
    # Index segmentation mask by position within image

    def __getitem__(self,idx):
        return self.Mask[idx]

    # get_limits
    #
    # Factory method to find bounding boxes for all segments
    #
    # Returns:
    #       List of limits for bounding boxes [(xmin0,ymin0,xmax0,ymax0),(xmin1,ymin1,xmax1,ymax1),...]

    def get_limits(self):
        Product = []
        for i in range(self.nx):
            for j in range(self.ny):
                k = int(self.Mask[i,j])
                while k>len(Product):
                    Product.append((self.nx,self.ny,-1,-1))
                xmin,ymin,xmax,ymax = Product[k-1]
                changed             = False

                # Establish whether (i,j) is outside current limits for box 'k'
                if i < xmin:
                    xmin    = i
                    changed = True
                if j < ymin:
                    ymin    = j
                    changed = True
                if xmax < i:
                    xmax    = i
                    changed = True
                if ymax < j:
                    ymax    = j
                    changed = True

                if changed:
                    Product[k-1] = (xmin,ymin,xmax,ymax)
        return Product

    #  apply
    #
    # Mask out all bits in image, except for selected segment
    #
    # Parameters:
    #     k         Number of segment, 1 based
    #     Limits    Defines bounding box (limits search: saves time, but does not affect result).
    #     Greys     Image to be masked - one colour only (greyscale)

    def apply(self,k,Limits,Greys):
        i0,j0,i1,j1 = Limits
        Product      = zeros((i1-i0,j1-j0))
        for i in range(i0,i1):
            for j in range(j0,j1):
                if self.Mask[i,j] == k:
                    Product[i-i0,j-j0] = Greys[i,j]
        return Product

    # save
    #
    # Save mask to specified file

    def save(self,file_name):
        save(file_name,self.Mask)

# restrict
#
# Used to restrict training data to specified labels
#
# Parameters:
#     Training   List of immage_ids for data
#     Labels     We want to restrict to image_ids that match these labels
#     multiple   Determines whether image_ids with multiple labels are to be in included (subject to at least one matching)

def restrict(Training,Labels,multiple=False):
    # should_include
    #
    # Verify that image_label matches allowed list
    #
    # Parameters:
    #     image_labels  List of labels for one image
    #
    # Returns: True iff 1. image_label is in our list Labels, AND
    #                   2. EITHER image has only one label OR multiple labels are allwed

    def should_include(image_labels):
        return (len(set(image_labels)& set(Labels))>0) and (multiple or len(image_labels)==1)
    return {image_id: image_labels for image_id,image_labels in Training.items() if should_include(image_labels)}



# get_dist
#
# Compute  Euclidean distance between two points
#
# Parameters:
#     pt0    One point
#     pt1    T'other point

def get_dist(pt0,pt1):
    x0,y0 = pt0
    x1,y1 = pt1
    return sqrt((x1-x0)**2 + (y1-y0)**2)


# get_centroid
#
# Determine centroids of a list of points
#
# Parameters:
#     Points

def get_centroid(Points):
    return (mean([x for x,_ in Points]),
            mean([y for _,y in Points]))

# DPmeans
#
# A generator to find clusters using Dirichlet process means
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
#     randomize    Randomize order of observations prior to clustering

def DPmeans(Image,
            Lambda     = 4000,
            background = 0,
            delta      = 8,
            randomize  = False,
            dist       = get_dist
            ):

    # extract_one_cluster
    #
    # Extract observations that should be assigned to specified cluster index

    def extract_one_cluster(index):
        return [Xs[i] for i in range(n) if Zs[i]==index]

    # has_converged
    #
    # Establish whether iterations have converged, i.e.: number of clusters hasn't changed since last iteration
    # and the centroids haven't moved by a distance of more than delta

    def has_converged(mu,mu0):
        return len(mu)==len(mu0) and all(dist(p1,p2)<delta for p1,p2 in zip(sorted(mu),sorted(mu0)))

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
    if randomize:
        shuffle(Xs)

    while True:
        for i in range(n):
            D = [dist(Xs[i],mu[c]) for c in range(k)]
            c = argmin(D)
            if D[c] > Lambda:     # Create new cluster
                Zs[i]     = k
                k         += 1
                mu.append(Xs[i])
            else:                  # Assign point to closest cluster
                if Zs[i] != c:
                    Zs[i] = c

        L  =  [extract_one_cluster(c) for c in range(k)]

        L1  = [c for c in L if len(c)>0]   # Eliminate empty clusters, e.g. 0a7e47d2-bbb2-11e8-b2ba-ac1f6b6435d0
        k   = len(L1)

        mu0 = mu[:]                        # Need copy to determine whether centroids have moved
        mu  = [get_centroid(Xs) for Xs in L1]
        yield has_converged(mu,mu0),k,L1,mu,Xs,Zs


# remove_isolated_centroids
#
# Remove centroids that have fewer than a specified number of points
#
# Parameters:
#     L
#     mu
#     cutoff

def remove_isolated_centroids(L,mu,cutoff = 20):
    mu = [m for m,cluster in zip(mu,L) if len(cluster) > cutoff]
    L  = [cluster for cluster in L if len(cluster) > cutoff]
    return len(L),mu,L

# get_image_file_name
#
# Construct actual file name for saving images
#
# Parameters:
#     image_id   Identifies image
#     figs       Path name for saved images
#     labels     Prepend labels to image)id when we save
#     sep        Separator character for multiple labals

def get_image_file_name(image_id,
                        figs   = '.',
                        labels = [],
                        sep    = '.'):
    return join(figs,f'{sep.join([str(label) for label in labels])}{sep}{image_id}.png')

# merge_greedy_centroids
#
# Used to merge centroids that appear to have divided on cluster between them
#
# Parameters:
#     k
#     L
#     mu
#     delta

def merge_greedy_centroids(k,L,mu,delta=1,dist=get_dist):
    # binary_search
    #
    # Find the nearest match to a specified value within a sorted list
    #
    # Parameters:
    #     x   The value to be found
    #     Xs The list
    #
    # Returns:
    #    Index of best match, or None

    def binary_search(x,Xs):
        if x<Xs[0] or x>Xs[-1]: return None
        low  = 0
        high = len(Xs)-1
        while low < high-1:
            mid = (low+high)//2
            if Xs[mid]<x:
                low = mid
            else:
                high = mid
        return mid

    # found
    #
    # Find point within a list

    def found(pt,L):
        Xs  = [x for x,_ in L]
        mid = binary_search(pt[0],Xs)
        if mid==None: return False
        i0 = mid
        while i0>0 and pt[0]-Xs[i0]<delta:
            i0-=1
        i1 = mid
        while i1<len(Xs) and Xs[i1]-pt[0]<delta:
            i1+=1
        for i in range(i0,i1):
            if dist(pt,L[i]) < delta: return True
        return False

    # create_merges
    #
    # Create lists of elements to be merged or skipped
    # Each pair of elements (a,b) has the property a>b
    # All data will go into cluster b, so a will need to
    # be skipped

    def create_merges(merge_pairs):
        merges = {}
        skips  = set()
        for a,b in merge_pairs:
            assert a>b
            skips.add(a)
            if not b in merges:
                merges[b] = [a]
            else:
                merges[b].append(a)
            if a in merges:
                merges[b].append(merges[a])
                del merges[a]
        return merges,skips

    # can_merge
    #
    # Determines whether two clusters can be merged

    def can_merge(i,j):
        mu_i  = mu[i]
        mu_j = mu[j]
        while delta < dist(mu_i,mu_j):
            mid = get_centroid([mu_i,mu_j])
            if found(mid,L[i]):
                mu_i = mid
            elif found(mid,L[j]):
                mu_j = mid
            else:
                return False
        return True

    L            = [sorted(l,key=lambda x:x[0]) for l in L]
    merges,skips = create_merges([(i,j) for i in range(1,k) for j in range(i) if can_merge(i,j)])

    if len(merges)==0:
        return len(L),mu,L
    else:
        L1  = []
        mu1 = []
        for i in range(len(L)):
            if i in skips: continue
            if i in merges:
                Merged = L[i][:]
                for j in merges[i]:
                    Merged = Merged + L[j]
                L1.append(Merged)
                mu1.append(get_centroid(Merged))
            else:
                L1.append(L[i])
                mu1.append(mu[i])
        return len(L1),mu1,L1

def show_clusters_and_centroids(Image,
                                L       = [],
                                mu      = [],
                                axis    = None,
                                colours = [],
                                title   = ''):
    Image.show(axis=axis)
    for l in range(len(L)):
        axis.scatter([x for x,_ in L[l]],[y for _,y in L[l]],c=colours[l],s=1)
    axis.scatter([x for x,_ in mu],[y for _,y in mu],
                     marker = 'x',
                     c      = colours[len(L)+1],
                     s      = 10)
    axis.set_title(title)


# segment
#
# Read slides and segment image: start with Dirichlet process means, then postprocess to remove phantom cells.
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
#     randomize    Randomize order of observations prior to clustering

def segment(image_id             = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0',
            path                 = join(environ['DATA'],'hpa-scc'),
            image_set            = 'train512x512',
            figs                 = '.',
            Descriptions         = [],
            Training             = [],
            Lambda               = 8000,
            background           = 0,
            delta                = 64,
            XKCD                 = [],
            N                    = 100,
            segments             = '.',
            min_size_for_cluster = 0,
            merge_greedy         = False,
            randomize            = False,
            details              = False,
            dist                 = get_dist):

    Image = Image4(image_id  = image_id,
                   path      = path,
                   image_set = image_set)
    seq = None
    for seq,(converged,k,L,mu,Xs,Zs) in enumerate(DPmeans(Image.get()[:,:,BLUE],
                                                          Lambda     = Lambda,
                                                          background = background,
                                                          delta      = delta,
                                                          randomize  = randomize,
                                                          dist       = dist)):
        if converged: break

        if seq>N:
            print (f'Failed to converge in {N} steps')
            break


    k1,mu1,L1 = remove_isolated_centroids(L,mu) if min_size_for_cluster>0 else (k,mu,L)
    k2,mu2,L2 = merge_greedy_centroids(k1,L1,mu1,dist=dist) if merge_greedy else (k1,L1,mu1)
    mask      = SegmentationMask(Image,mu,dist=dist)
    mask.save(join(segments,f'{image_id}.npy'))

    if details:
        fig = figure(figsize=(27,18))
        axs = fig.subplots(nrows = 1, ncols = 2)

        show_clusters_and_centroids(Image,
                                    L       = L,
                                    mu      = mu,
                                    axis    = axs[0],
                                    colours = [colour for colour in create_xkcd_colours(filter = lambda R,G,B:R<192 and max(R,G,B)>32)][::-1],
                                    title   = f'Lambda={Lambda}, k={k}, iteration={seq}')

        Image.show(axis=axs[0],channels=[RED])

        Image.show(axis=axs[1],channels=[BLUE,RED])

        axs[1].scatter([x for x,_ in mu],[y for _,y in mu],
                         marker = 'x',
                         c      = XKCD[len(L)+1],
                         s      = 10)

    fig                = figure(figsize=(27,18))
    axs                = fig.subplots(nrows = 3, ncols = 3)
    # Head up plot with image_id and labels

    fig.suptitle(f'{image_id}: {", ".join([Descriptions[label] for label in Training[image_id]])}')
    axs[0][0].set_ylabel(f'Segmentation',
                         rotation = 'horizontal',
                         labelpad = 35)

    show_clusters_and_centroids(Image,
                                L       = L,
                                mu      = mu,
                                axis    = axs[0,0],
                                colours = XKCD,
                                title   = f'Lambda={Lambda}, k={k}, iteration={seq}')

    show_clusters_and_centroids(Image,
                                L       = L2,
                                mu      = mu2,
                                axis    = axs[0,1],
                                colours = XKCD,
                                title   = f'{IMAGE_LEVEL_LABELS[BLUE]}: k={k}')

    # Show bounding boxes

    Limits     = mask.get_limits()

    cmap       = ListedColormap(XKCD[:len(Limits)])
    mask_image = axs[0,2].imshow(mask.Mask.transpose(),
                                 interpolation = 'nearest',
                                 origin        = 'lower',
                                 cmap          = cmap)
    fig.colorbar(mask_image,ax=axs[0,2])
    for i,(x0,y0,x1,y1) in enumerate(Limits):
        axs[0,2].add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                     linewidth = 1,
                                     edgecolor = XKCD[len(Limits) + 1 +i],
                                     facecolor = 'none'))
    axs[0,2].set_title('Masks and Bounding boxes')
    x0, x1 = axs[0,0].get_xlim()
    y0, y1 = axs[0,0].get_ylim()
    axs[0,2].set_xlim(x0, x1)
    axs[0,2].set_ylim(y0, y1)

    # Show distribution of each filter compared to Nuclei channels
    axs[1][0].set_ylabel(f'{IMAGE_LEVEL_LABELS[BLUE]}+',rotation='horizontal', labelpad=40)
    for channel,ax in zip([RED,YELLOW,GREEN],
                          [axs[1,0],axs[1,1],axs[1,2]]):
        Image.show(axis=ax,channels=[BLUE,channel])
        ax.set_title(f'{IMAGE_LEVEL_LABELS[channel]}', y = 0.95) # Tuned to minimize overwriting

    # Show distribution of protein vs each other filter

    # voronoi = Voronoi(mu)
    axs[2][0].set_ylabel(f'{IMAGE_LEVEL_LABELS[GREEN]}+',
                         rotation = 'horizontal',
                         labelpad = 40)
    for channel,ax in zip([RED,YELLOW,BLUE],
                          [axs[2,0],axs[2,1],axs[2,2]]):
        # voronoi_plot_2d(voronoi,
                        # ax            = ax,
                        # show_vertices = False,
                        # show_points   = False,
                        # line_colors   = 'orange')
        Image.show(axis     = ax,
                   channels = [channel,GREEN],
                   actuals  = [BLUE,RED])
        ax.set_title(f'{IMAGE_LEVEL_LABELS[channel]}', y = 0.95) # Tuned to minimize overwriting of text


    savefig(get_image_file_name(
                image_id,
                figs   = figs,
                labels = Training[image_id]),
            dpi         = args.dpi,
            bbox_inches = 'tight')

    return fig,seq



if __name__=='__main__':
    filterwarnings('error', "Mean of empty slice.")
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
                        default = 4000, # Tested with  0a7e47d2-bbb2-11e8-b2ba-ac1f6b6435d0
                        help    = 'Penalty to discourage creating new clusters')
    parser.add_argument('--background',
                        type    = float,
                        default = 0.05,
                        help    = 'Threshold for blue pixels. We consider pixels only if they exceed this value.')
    parser.add_argument('--delta',
                        type    = float,
                        default = 1,
                        help    = 'If no centroids have shifted by more than this limit, we deem iterations to have converged')
    parser.add_argument('--N',
                        default = 75, # Processed 1307 slides, and the most iteration I have observed was 57.
                        type    = int,
                        help    = 'Maximum number of iterations for DPmeans')
    parser.add_argument('--singleton',
                        default = False,
                        action = 'store_true',
                        help   = 'Process singletons only')
    parser.add_argument('--worklist',
                        default = None,
                        help   = 'List of IDs to be processed')
    parser.add_argument('--segments',
                        default = './segments',
                        help    = 'Identifies where to store cell masks')
    parser.add_argument('--min_size_for_cluster',
                        default = 5,
                        type    = int,
                        help    = 'Used to delete clusters that are too small')
    parser.add_argument('--merge',
                        default = False,
                        action  = 'store_true',
                        help    = 'Merge clusters if they appear to have been split')
    parser.add_argument('--randomize',
                        default = False,
                        action  = 'store_true',
                        help    = 'Randomize observations prior to clustering')
    parser.add_argument('--detail',
                        default = False,
                        action  = 'store_true',
                        help    = 'Show a second, more detailed, plot')
    args = parser.parse_args()
    if args.randomize:
        set_random_seed(specified_seed=args.seed)

    with Timer(),Logger('dirichlet') as logger:
        XKCD         = [colour for colour in create_xkcd_colours()][::-1]
        Descriptions = read_descriptions('descriptions.csv')
        Training     = restrict(read_training_expectations(path=args.path),
                                Labels   = args.labels,
                                multiple = args.multiple or args.image_id!=None)

        if args.worklist != None:
            for image_id in read_worklist(args.worklist):
                fig = None
                try:
                    fig,seq = segment(image_id             = image_id,
                                      path                 = args.path,
                                      image_set            = args.image_set,
                                      figs                 = args.figs,
                                      Descriptions         = Descriptions,
                                      Training             = Training,
                                      background           = args.background,
                                      XKCD                 = XKCD,
                                      N                    = args.N,
                                      delta                = args.delta,
                                      Lambda               = args.Lambda,
                                      segments             = args.segments,
                                      min_size_for_cluster = args.min_size_for_cluster,
                                      merge_greedy         = args.merge,
                                      randomize            = args.randomize,
                                      details              = args.detail)
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
            _,seq = segment(image_id             = args.image_id,
                            path                 = args.path,
                            image_set            = args.image_set,
                            figs                 = args.figs,
                            Descriptions         = Descriptions,
                            Training             = Training,
                            background           = args.background,
                            XKCD                 = XKCD,
                            N                    = args.N,
                            delta                = args.delta,
                            Lambda               = args.Lambda,
                            segments             = args.segments,
                            min_size_for_cluster = args.min_size_for_cluster,
                            merge_greedy         = args.merge,
                            randomize            = args.randomize,
                            details              = args.detail)

            print (f'Segmented {args.image_id},{seq}')
            logger.log(f'{args.image_id},{seq}')

    if args.show:
        show()
