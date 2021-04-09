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

import csv
from dirichlet import Image4
import math
import matplotlib.pyplot as plt
import numpy as np
from   os.path import join
import sys
from utils import Timer


def get_centroids(image_id,path='./segments'):
     with open (join(path,f'{image_id}.csv')) as file:
          return [tuple([float(s) for s in row]) for row in csv.reader(file)]

def create_mask(Image,Centroids=[]):
     Mask  = np.zeros((Image.ny,Image.ny))
     for i in range(Image.nx):
          for j in range(Image.ny):
               Mask[i,j]    = None
               max_distance = sys.float_info.max
               for k in range(len(Centroids)):
                    i0,j0 = Centroids[k]
                    distance = (i-i0)*(i-i0) + (j-j0)*(j-j0)
                    if distance<max_distance:
                         Mask[i,j]    = k
                         max_distance = distance
     return Mask

image_id  = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'


with Timer():
     Centroids = get_centroids(image_id)
     Image     = Image4(image_id=image_id)
     Mask  = create_mask(Image, Centroids=Centroids)
     nrows     = math.isqrt(len(Centroids))
     ncols     = nrows
     while nrows*ncols<len(Centroids):
          ncols+= 1

     fig                = plt.figure(figsize=(27,18))
     axs                = fig.subplots(nrows = nrows, ncols = ncols)
     for i in range(nrows):
          for j in range(ncols):
               k = 3*i+j
               axs[i][j].imshow(Image.get_segment(Mask     = Mask,
                                                  selector = k,
                                                  channels = [0,1,2,3]))

plt.show()
