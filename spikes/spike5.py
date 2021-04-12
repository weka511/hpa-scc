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
#  Figure out how to read segmentation masks
#
# Note this remark at https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
# "I try to avoid cv2, it swaps dimensions and loads in BGR channel format.."

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def decode_colours(r,g,b):
    def decode_channel(c):
        return int(256*c)
    return (decode_channel(r)*256 + decode_channel(g))*256 + decode_channel(b)

fig = plt.figure(figsize=(10,10))
for key,value in fig.canvas.get_supported_filetypes().items():
    print(key,value)

image_id = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'

path = f'./segments/{image_id}_mask.tif'

# raw = np.fromfile(path, dtype='int8', sep="")
# raw = plt.imread(path,format='raw')
raw = plt.imread(path,format='tif')
print (raw.shape)
# A = np.reshape(raw, (1000, 1000, 4))

# img_plt =plt.imread(path)

# img_np = np.array(img_plt)

# img_cv = cv2.imread(path)



axs = fig.subplots(nrows = 2, ncols = 2)
axs[0,0].imshow(raw)
# axs[0,1].imshow(img_np)
# axs[1,0].imshow(img_cv)

# res = cv2.resize(img_cv, dsize=(512,512), interpolation=cv2.INTER_CUBIC) # BGR
# res2 = np.zeros_like(res)   #RGB
# res2[:,:,0] = res[:,:,2]
# res2[:,:,1] = res[:,:,1]
# res2[:,:,2] = res[:,:,0]
# axs[1,1].imshow(res2)
# axs[1,1].set_title(f'{res2.shape}')
# fig.suptitle(f'{image_id}')

# nx,ny,_ = raw.shape
# counts = []
# for i in range(nx):
    # for j in range(ny):
        # n = 1+ decode_colours(raw[i,j,0],raw[i,j,1],raw[i,j,2])
        # while len(counts)<n+1:
            # counts.append(0)
        # counts[n]+=1
# axs[0,1].bar(range(len(counts)),counts)
plt.show()
