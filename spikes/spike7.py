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
import numpy as np
from   os.path import join
import sys
from   utils import Timer

image_id  = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'


with Timer():
     Image  = Image4(image_id=image_id)
     mask   = Mask.Load(join('./segments',f'{image_id}.npy'))
     Limits = mask.get_limits()
     nrows  = math.isqrt(len(Limits))
     ncols  = nrows
     while nrows*ncols<len(Limits):
          ncols+= 1

     fig                = plt.figure(figsize=(27,18))
     axs                = fig.subplots(nrows = nrows, ncols = ncols)
     for i in range(nrows):
          for j in range(ncols):
               k = 3*i+j + 1
               axs[i][j].imshow(Image.get_segment(Mask     = mask,
                                                  selector = k,
                                                  channels = [0,1,2,3]))

plt.show()
