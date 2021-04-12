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
#
# See https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
# https://bic-berkeley.github.io/psych-214-fall-2016/resampling_with_ndimage.html
import math
import matplotlib.pyplot as plt
from   matplotlib.image  import imread
import numpy as np
from   os.path import join
from   scipy.ndimage import affine_transform
import sys

image_id  = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'
path_name = join (r'D:\data\hpa-scc\train512x512',f'{image_id}_blue.png')


def rotate(theta):
     cos_t = np.cos(theta)
     sin_t = np.sin(theta)
     return np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])
def get_centroid(G):
     x = 0
     y = 0
     total = 0
     m,n = G.shape
     for i in range(m):
          for j in range(n):
               x += G[i,j]*i
               y += G[i,j]*j
               total += G[i,j]
     return (x/total,y/ total)

fig               = plt.figure(figsize=(27,18))
axs               = fig.subplots(nrows = 2, ncols = 5)

Greys              = imread(path_name)
Greys[0:10,0:10]   = np.amax(Greys)                 # Add marker for origin
Greys[0:10,-10:-1] = 0.5*np.amax(Greys)           # Add marker for y axis
axs[0][0].imshow(Greys.transpose())
axs[0][0].set_title('Original')

G3 = Greys[270:312,240:340]
x0,y0 = get_centroid(G3)

axs[0][1].imshow(G3.transpose())
axs[0][1].plot(x0,y0,'or')
axs[0][1].set_title('Segment')

mult = 1
for row in range(2):
     for col in range(0 if row>0 else 2,5):
          c_in      = 0.5*np.array(G3.shape)
          c_out     = np.array((64,64))
          transform = rotate(mult*np.pi/8)
          offset    = c_in-c_out.dot(transform)
          K5        = affine_transform(G3, transform.T, offset=offset, order=1,output_shape=(128,128),cval=np.amax(Greys))
          axs[row][col].imshow(K5.transpose())
          axs[row][col].set_title('rotated and offset')
          mult += 1

plt.show()
