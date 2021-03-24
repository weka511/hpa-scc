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
from matplotlib.pyplot import hist, show, figure, savefig
from matplotlib.image  import imread
from numpy             import zeros, mean, std, argmin
from os                import environ
from os.path           import join

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
# read_image
#
# Read set of images (keeping Yellow separate)
#
# Parameters:
#    path            Location of data
#    image_set       Identifies high or low res
#    image_id        Identified image

def read_image(path        = r'C:\data\hpa-scc',
               image_set   = 'train512512',
               image_id    = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'):
    Image = None
    for channel in range(NCHANNELS):
        image_mono = imread(join(path,
                                 image_set,
                                 f'{image_id}_{COLOUR_NAMES[channel]}.png'))
        if channel==0:
            nx,ny          = image_mono.shape
            Image          = zeros((nx,ny,NCHANNELS))

        Image[:,:,channel] = image_mono

    return Image

def DPmeans(Image,Lambda=2000,background=0,delta=64):
    def get_dist2(p0,p1):
        x0,y0 = p0
        x1,y1 = p1
        return (x1-x0)**2 + (y1-y0)**2

    def get_centroid(Points):
        return (mean([x for x,_ in Points]),
                mean([y for _,y in Points]))

    def generate_cluster(c):
        return [Xs[i] for i in range(n) if Zs[i]==c]

    def has_converged(mu,mu0):
        if len(mu)!=len(mu0):
            return False
        else:
            return all(get_dist2(p1,p2)<delta for p1,p2 in zip(sorted(mu),sorted(mu0)))

    nx,ny = Image.shape
    Xs        = [(i,j) for i in range(nx) for j in range(ny) if Image[i,j]>background]
    n         = len(Xs)
    k         = 1
    L         = [Xs]
    mu        = [get_centroid(Xs)]
    Zs        = [0 for _ in Xs]
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

if __name__=='__main__':
    parser = ArgumentParser('Cluster HPA data')
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
                        default = True,
                        action  = 'store_true',
                        help    = 'Display images')
    parser.add_argument('--dpi',
                        default = 300,
                        type    = int,
                        help    = 'Resolution for saving images')
    args = parser.parse_args()
    Image = read_image(image_id  = args.image_id,
                       path      = args.path,
                       image_set = args.image_set)

    for l,(converged,k,L,mu,Xs,Zs) in enumerate(DPmeans(Image[:,:,BLUE])):
        if converged: break

    fig   = figure(figsize=(20,20))
    nrows = 4
    ncols = 4
    axs   = fig.subplots(nrows = 2, ncols = 2)
    axs[0,0].scatter([x for x,_ in Xs],[y for _,y in Xs],c='b',s=1,alpha=0.5)
    axs[0,0].scatter([x for x,_ in mu],[y for _,y in mu],c='r',s=1,alpha=0.5)
    axs[0,0].set_title(f'k={k},l={l}')
    fig.suptitle(f'{args.image_id}')
    savefig(join(args.figs,f'{args.image_id}_dirichlet'), dpi=args.dpi, bbox_inches='tight')
    if args.show:
        show()
