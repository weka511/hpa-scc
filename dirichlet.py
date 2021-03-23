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
from matplotlib.pyplot import hist, show, figure
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

def DPmeans(Image,Lambda=200,threshold=0):
    def get_centroid(Points):
        return (mean([x for x,_ in Points]),
                mean([y for _,y in Points]))
    def generate_cluster(c):
        return [Xs[i] for i in range(n) if Zs[i]==c]

    nx,ny = Image.shape
    Xs    = [(i,j) for i in range(nx) for j in range(ny) if Image[i,j]>threshold]
    n     = len(Xs)
    k     = 1
    L     = [Xs]
    mu    = [get_centroid(Xs)]
    Zs    = [0 for _ in Xs]
    fig          = figure(figsize=(20,20))
    axs          = fig.subplots(nrows = 4, ncols = 4)
    for l in range(10):
        for i in range(n):
            x,y    = Xs[i]
            d = []
            for c in range(k):
                x0,y0  = mu[c]
                d.append((x-x0)**2 + (y-y0)**2)
            c = argmin(d)
            if d[c] > Lambda:
                Zs[i] = k
                k += 1
                mu.append(Xs[i])
            else:
                Zs[i] = c

        L = [generate_cluster(c) for c in range(k)]
        mu = [get_centroid(Xs) for Xs in L]
        axs[l//4,l%4].scatter([x for x,_ in mu],[y for _,y in mu])

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
    args = parser.parse_args()
    Image = read_image(image_id  = args.image_id,
                       path      = args.path,
                       image_set = args.image_set)
    DPmeans(Image[:,:,BLUE])
    show()
