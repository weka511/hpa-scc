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
from   matplotlib       import cm
from   os.path          import join,basename
import numpy            as np
from   argparse         import ArgumentParser
from   random           import choice
from   time             import time

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


def otsu(Image,nx,ny,tolerance=0.0001,N=50):
    def get_icv(threshold):
        P1  = [blue for blue in Blues if blue<threshold]
        P2 = [blue for blue in Blues if blue>threshold]        
        var1 = np.var(P1)
        var2 = np.var(P2)
        return (len(P1)*var1 + len(P2)*var2)/(len(P1) + len(P2))    
    Blues = [Image[i,j,BLUE] for i in range(nx) for j in range(ny)]
    n,bins,_ = axs[0,1].hist(Blues)
    threshold1 = bins[1]
    threshold2 = bins[-2]
    icv1       = get_icv(threshold1)
    icv2       = get_icv(threshold2)
    ICVs      = [icv1,icv2]
    for _ in range(N):
        if abs(icv1-icv2)<tolerance: break
        threshold_mid = 0.5*(threshold1 + threshold2)
        icv_mid       = get_icv(threshold_mid)
        if icv1<icv2:
            threshold2 = threshold_mid
            icv2       = icv_mid
        else:
            threshold1 = threshold_mid
            icv1       = icv_mid
        ICVs.append(icv_mid)
    axs[1,0].scatter(range(len(ICVs)),ICVs)
    return threshold_mid,icv_mid

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Segment HPA data using Otsu\'s algorithm')
    parser.add_argument('--path',                default=r'C:\data\hpa-scc')
    parser.add_argument('--image_set',           default = 'train512512')
    parser.add_argument('--image_id',            default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0')
    parser.add_argument('--show',                default=False, action='store_true')
 
    args     = parser.parse_args()
    
    Training = read_training_expectations(path=args.path)
    Image    = read_image(path=args.path,image_id=args.image_id,image_set=args.image_set)
    
    fig      = plt.figure(figsize=(20,20))
    axs      = fig.subplots(2, 2) 
    nx,ny,_  = Image.shape
    im       = axs[0,0].imshow(Image[:,:,BLUE],cmap=cm.get_cmap('Blues'))
    axs[0,0].axes.xaxis.set_ticks([])
    axs[0,0].axes.yaxis.set_ticks([]) 
    fig.colorbar(im, ax=axs[0,0], orientation='vertical')
    threshold_mid,icv_mid=otsu(Image,nx,ny)
    Partitioned = np.zeros((nx,ny,3))
    for i in range(nx):
        for j in range(ny):
            k = BLUE if Image[i,j,BLUE]<threshold_mid else RED
            Partitioned[i,j,k] = Image[i,j,BLUE]
    axs[1,1].imshow(Partitioned)        
    mylabels =  '+'.join([Descriptions[label] for label in Training[image_id]])      
    fig.suptitle(f'{args.image_id}: {mylabels}')
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    plt.savefig(f'{basename(__file__).split(".")[0]}.png')
    if args.show:
        plt.show()