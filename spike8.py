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

with Timer(): # https://bic-berkeley.github.io/psych-214-fall-2016/resampling_with_ndimage.html
     fig   = plt.figure(figsize=(27,18))
     axs   = fig.subplots(nrows = 2, ncols = 4)
     I           = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
     Greys       = imread(path_name)
     Greys[0:10,0:10] = np.amax(Greys)                 # Add marker for origin
     Greys[0:10,-10:-1] = 0.5*np.amax(Greys)           # Add marker for y axis
     axs[0][0].imshow(Greys.transpose())
     axs[0][0].set_title('Original')

     translation1 = [0,0]
     K1           = affine_transform(Greys, I, offset=translation1, order=1,output_shape=(256,256))
     axs[0][1].imshow(K1.transpose())
     axs[0][1].set_title('Down sampled')

     translation2 = [300,500]
     K2           = affine_transform(Greys, I, offset=translation2, order=1,output_shape=(256,256))
     axs[0][2].imshow(K2.transpose())
     axs[0][2].set_title('Down sampled & shifted')

     # M           = z_rotmat(np.pi/8)
     # translation = [0,0,0]
     # K           = affine_transform(Greys, M, translation, order=1,output_shape=(256,256))
     # print (K.shape)

     G3 = Greys[270:315,225:350]
     axs[0][3].imshow(G3.transpose())
     axs[0][3].set_title('Segment')

     K3  = affine_transform(G3, I, offset=[0,0,0], order=1,output_shape=(128,128),cval=np.amax(Greys))
     axs[1][0].imshow(K3.transpose())
     axs[1][0].set_title('Segmented and resampled')

     M4           = z_rotmat(np.pi/8)
     K4  = affine_transform(G3, M4, offset=[0,0,0], order=1,output_shape=(128,128),cval=np.amax(Greys))
     axs[1][1].imshow(K4.transpose())
     axs[1][1].set_title('Rotated about origin')
     c_in   = 0.5*np.array(G3.shape)
     c_out  = np.array(G3.shape)
     transform=np.array([[np.cos(np.pi/8),-np.sin(np.pi/8)],[np.sin(np.pi/8),np.cos(np.pi/8)]]).dot(np.diag(([1,1])))
     offset = c_in-c_out.dot(transform)
     K5  = affine_transform(G3, transform, offset=offset, order=1,output_shape=(128,128),cval=np.amax(Greys))
     axs[1][2].imshow(K5.transpose())
     axs[1][2].set_title('Segmented and resampled')

plt.show()
