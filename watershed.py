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

def get_neighbours(x,y,nx=256,ny=256,delta=1):
    candidates = [(x+dx,y+dy) for dx in [-delta,0,delta] for dy in [-delta,0,delta] if dx !=0 or dy !=0]
    return [(x1,y1) for x1,y1 in candidates if 0<=x1 and x1<nx and 0<=y1 and y1<ny]

def create_admissable(Candidates,m=10,Delta=10000):
    def get_distance(P1,P2):
        x1,y1,_ = P1
        x2,y2,_ = P2
        return (x1-x2)**2 + (y1-y2)**2
    
    admissable = False
    while not admissable:
        Possibities = [choice(Candidates) for _ in range(m)]
        for i in range(m):
            for j in range(i+1,m):
                if get_distance(Possibities[i],Possibities[2])<Delta:
                    admissable = False
        admissable = True
        
    for configuration in Possibities:
        yield configuration
        
def explore(Image,
            threshold1 = 0.01, 
            threshold2 = 0.5,
            nx         = 512,
            ny         = 512,
            delta      = 16,
            axs        = None,
            fmt        = 'r-',
            n          = 5000,
            m          = 32):
    Candidates = [(i,j,Image[i,j,BLUE]) for i in range(nx) for j in range(ny) if Image[i,j,BLUE]<threshold1]
    for x0,y0,blue in create_admissable(Candidates,m=m):
        x1,y1        = x0,y0
        axs.scatter(y0,x0,c='r',s=100,marker='x')
        for i in range(n):
            Neighbours = get_neighbours(x0,y0,nx,ny,delta=delta)
            Blues      = sorted([(x,y,Image[x,y,BLUE]) for x,y in Neighbours],
                                key = lambda x:x[2],
                                reverse=True)
            Increasing = [(x,y,b) for x,y,b in Blues if b>blue]
            Same       = [(x,y,b) for x,y,b in Blues if b==blue]
            Decreasing = [(x,y,b) for x,y,b in Blues if b<blue]
            if len(Increasing)>0:
                x1,y1,blue = choice(Increasing)
            elif len(Same)>0:
                x1,y1,blue = choice(Same)
            else:
                x1,y1,blue = choice(Decreasing)
            while blue>threshold2 and delta>1:
                delta = 1
            axs.plot([y0,y1], [x0,x1], fmt)
            x0,y0 = x1,y1
        axs.scatter(y1,x1,c='r',s=100,marker='o')

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Segment HPA data using Watershed algorithm')
    parser.add_argument('--path',                default=r'C:\data\hpa-scc')
    parser.add_argument('--image_set',           default = 'train512512')
    parser.add_argument('--image_id',            default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0')
    parser.add_argument('--n',         type=int, default = 5000)
    parser.add_argument('--m',         type=int, default = 32)
    parser.add_argument('--show',                default=False, action='store_true')
 
    args     = parser.parse_args()
    
    Training = read_training_expectations(path=args.path)
    Image    = read_image(path=args.path,image_id=args.image_id,image_set=args.image_set)
    
    fig      = plt.figure(figsize=(20,20))
    axs      = fig.subplots(1, 1) 
    nx,ny,_  = Image.shape
    im       = axs.imshow(Image[:,:,BLUE],cmap=cm.get_cmap('Blues'))
    axs.axes.xaxis.set_ticks([])
    axs.axes.yaxis.set_ticks([]) 
    fig.colorbar(im, ax=axs, orientation='vertical') 
    
    explore(Image,axs=axs,nx=nx,ny=ny, n=args.n, m=args.m)

    mylabels =  '+'.join([Descriptions[label] for label in Training[image_id]])      
    fig.suptitle(f'{args.image_id}: {mylabels}')
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    plt.savefig(f'{basename(__file__).split(".")[0]}.png')
    if args.show:
        plt.show()