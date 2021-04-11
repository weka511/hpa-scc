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
#
#  Figure out how to extract segmented images using saved Mask

from   dirichlet import Image4,Mask
import math
import matplotlib.pyplot as plt
from   matplotlib.image  import imread
import numpy as np
from   os.path import join
from   scipy.ndimage import affine_transform
import sys
from   utils import Timer

image_id  = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'
path_name = join (r'D:\data\hpa-scc\train512x512',f'{image_id}_blue.png')

def x_rotmat(theta):
     cos_t = np.cos(theta)
     sin_t = np.sin(theta)
     return np.array([[1, 0, 0],
                     [0, cos_t, -sin_t],
                     [0, sin_t, cos_t]])


def y_rotmat(theta):
     cos_t = np.cos(theta)
     sin_t = np.sin(theta)
     return np.array([[cos_t, 0, sin_t],
                     [0, 1, 0],
                     [-sin_t, 0, cos_t]])


def z_rotmat(theta):
     cos_t = np.cos(theta)
     sin_t = np.sin(theta)
     return np.array([[cos_t, -sin_t, 0],
                     [sin_t, cos_t, 0],
                     [0, 0, 1]])

with Timer():
     Greys       = imread(path_name)
     M           = z_rotmat(np.pi/8)
     translation = [0,0,0]
     K           = affine_transform(Greys, M, translation, order=1,output_shape=(256,256))
     print (K.shape)
     fig   = plt.figure(figsize=(27,18))
     axs   = fig.subplots(nrows = 2, ncols = 2)

     axs[0][0].imshow(Greys)
     axs[0][1].imshow(K)
plt.show()
