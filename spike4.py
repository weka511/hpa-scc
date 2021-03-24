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

from matplotlib.pyplot import show, figure, imshow, scatter
from matplotlib.image  import imread
from numpy             import zeros
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

class Image4(object):
    def __init__(self,
                 path        = r'd:\data\hpa-scc',
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

image = Image4()
blues = image.get()
for i in range(10):
    for j in range(10):
        blues[i,j]=1
imshow(blues,extent=[0,image.nx-1,0,image.ny-1],origin='lower')
scatter([0],[0],c='r')
scatter([image.nx-1],[image.ny-1],c='m')
show()
