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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

colours     = ['red',  'green', 'blue', 'yellow']
colour_maps = [plt.cm.Reds,plt.cm.Blues,plt.cm.Greens,plt.cm.Oranges]
meanings    = ['Microtubules', 'Nucleus', 'Protein/antibody', 'Endoplasmic reticulum']
fig, axs    = plt.subplots(2, 2)
path        = r'C:\data\hpa-scc\train512512'
image_id    = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'
image       = None

for i in range(2):
    for j in range(2):
        index      = 2*i+j
        file_name  = f'{image_id}_{colours[index]}.png'
        path_name  = os.path.join(path,file_name)
        image_mono = mpimg.imread(path_name)
        if index==0:
            nx,ny   = image_mono.shape
            image   = np.zeros((nx,ny,4))
        image[:,:,index] = image_mono
 
axs[0,0].imshow(image[:,:,[0,2,3]])
        
fig.suptitle(image_id)
plt.show()