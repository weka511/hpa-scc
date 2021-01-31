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
from   matplotlib.image import imread
from   matplotlib import cm
from   os.path import join
import numpy as np
import argparse

Descriptions = [
    'Nucleoplasm',
    'Nuclear membrane',
    'Nucleoli',
    'Nucleoli fibrillar center',
    'Nuclear speckles',
    'Nuclear bodies',
    'Endoplasmic reticulum',
    'Golgi apparatus',
    'Intermediate filaments',
    'Actin filaments',
    'Microtubules',
    'Mitotic spindle',
    'Centrosome',
    'Plasma membrane',
    'Mitochondria',
    'Aggresome',
    'Cytosol',
    'Vesicles and punctate cytosolic patterns',
    'Negative'
]

RED         = 0
GREEN       = 1
BLUE        = 2
YELLOW      = 3
colours     = ['red',  'green', 'blue', 'yellow']
meanings    = ['Microtubules', 'Nuclei channels', 'Protein/antibody', 'Endoplasmic reticulum channels']
image_id    = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'


def read_training_expectations(path=r'C:\data\hpa-scc',file_name='train.csv'):
    header    = True
    Training  = {}
    for line in open(join(path,file_name)):
        if header:
            header = False
            continue
        trimmed = line.strip().split(',')
        Training[trimmed[0]] =  [int(l) for l in trimmed[1].split('|')]
    return Training

def read_image(path        = r'C:\data\hpa-scc',
               image_set   = 'train512512',
               image_id    = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'):
    Image = None
    for i in range(2):
        for j in range(2):
            index      = 2*i+j
            file_name  = f'{image_id}_{colours[index]}.png'
            path_name  = join(path,image_set,file_name)
            image_mono = imread(path_name)
            if index==0:
                nx,ny   = image_mono.shape
                Image   = np.zeros((nx,ny,4))
            Image[:,:,index] = image_mono
    return Image        

def create_selection(Image,
                 Selector = [
                     [1,0,0,1],
                     [0,1,0,1],
                     [0,0,1,0]]):
    nx,ny,_ = Image.shape
    Product = np.zeros((nx,ny,3))
    Matrix  = np.array(Selector)
    for i in range(nx):
        for j in range(ny):
            for k in range(3):
                Product[i,j,k] = sum([Matrix[k,l] * Image[i,j,l] for l in range(4)])
  
    return Product

if __name__=='__main__':
    parser = argparse.ArgumentParser('Visualize HPA data')
    parser.add_argument('--path',      default=r'C:\data\hpa-scc')
    parser.add_argument('--image_set', default = 'train512512')
    parser.add_argument('--image_id',  default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0')
    args     = parser.parse_args()
    
    Training = read_training_expectations(path=args.path)
    Image    = read_image(path=args.path,image_id=args.image_id,image_set=args.image_set)
    
    fig = plt.figure(figsize=(20,20))
    axs = fig.subplots(2, 4)
    
    axs[0,0].imshow(create_selection(Image)) #(Image[:,:,[0,2,3]])
    axs[0,0].axes.xaxis.set_ticks([])
    axs[0,0].axes.yaxis.set_ticks([])
    nx,ny,_  = Image.shape
    for i in range(3):
        ImageR   = np.zeros((nx,ny,3))
        ImageR[:,:,i] = Image[:,:,i]
        axs[0,i+1].imshow(ImageR)
        axs[0,i+1].axes.xaxis.set_ticks([])
        axs[0,i+1].axes.yaxis.set_ticks([])

    ImageY   = np.zeros((nx,ny,3))
    ImageY[:,:,0] = Image[:,:,3]
    ImageY[:,:,1] = Image[:,:,3]
    axs[1,0].imshow(ImageY)
    axs[1,0].axes.xaxis.set_ticks([])
    axs[1,0].axes.yaxis.set_ticks([])
    
    jet = cm.get_cmap('jet')
    im = axs[1,1].imshow(Image[:,:,BLUE],cmap=jet)
    axs[1,1].axes.xaxis.set_ticks([])
    axs[1,1].axes.yaxis.set_ticks([]) 
    fig.colorbar(im, ax=axs[1,1], orientation='vertical') 
    mylabels =  '+'.join([Descriptions[label] for label in Training[image_id]])      
    fig.suptitle(f'{args.image_id}: {mylabels}')
    plt.show()