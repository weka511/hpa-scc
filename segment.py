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
# Segment images using Otsu's method

from   argparse          import ArgumentParser
from   matplotlib.pyplot import figure, show, cm, close
from   matplotlib.image  import imread
from   numpy             import zeros, array, var, mean, std, histogram
from   os                import remove
from   os.path           import join,basename
from   random            import sample
from   tempfile          import gettempdir
from   time              import time
from   uuid              import uuid4

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
                Image   = zeros((nx,ny,4))
            Image[:,:,index] = image_mono
    return Image        

def create_selection(Image,
                 Selector = [
                     [1,0,0,1],
                     [0,1,0,1],
                     [0,0,1,0]]):
    nx,ny,_ = Image.shape
    Product = zeros((nx,ny,3))
    Matrix  = array(Selector)
    for i in range(nx):
        for j in range(ny):
            for k in range(3):
                Product[i,j,k] = sum([Matrix[k,l] * Image[i,j,l] for l in range(4)])
  
    return Product

def otsu(Image,nx,ny,tolerance=0.0001,N=50):
    def get_icv(threshold):
        P1   = [blue for blue in Blues if blue<threshold]
        P2   = [blue for blue in Blues if blue>threshold]        
        var1 = var(P1)
        var2 = var(P2)
        return (len(P1)*var1 + len(P2)*var2)/(len(P1) + len(P2))  
    
    Blues = [Image[i,j,BLUE] for i in range(nx) for j in range(ny)]
    n, bins = histogram(Blues)

    threshold1 = bins[1]
    threshold2 = bins[-2]
    icv1       = get_icv(threshold1)
    icv2       = get_icv(threshold2)
    ICVs      =  [icv1,icv2]
    Thresholds = [threshold1,threshold2]
    for _ in range(N):
        threshold_mid = 0.5*(threshold1 + threshold2)
        icv_mid       = get_icv(threshold_mid)
        if abs(icv1-icv2)<tolerance: break
        if icv1<icv2:
            threshold2 = threshold_mid
            icv2       = icv_mid
        else:
            threshold1 = threshold_mid
            icv1       = icv_mid
        ICVs.append(icv_mid)
        Thresholds.append(threshold_mid)
         
    return Thresholds,ICVs,n, bins


def generate8components(Image,threshold=0.5,nx=512,ny=512,deltas=[-1,0,1]):
    def find_first_ripe():
        for i in range(nx):
            for j in range(ny):
                if Open[i,j] and not Closed[i,j]:
                    return i,j
    
    Open      = zeros((nx,ny),dtype=bool) # Set of points that are potentially in a component
    
    for i in range(nx):
        for j in range(ny):
            if Image[i,j,BLUE]>threshold:
                Open[i,j] = True
                
    Closed    = zeros((nx,ny),dtype=bool)  # Set of points that have been processed already
    
    Ripe = set([find_first_ripe()])        # Here is where we start building a component
    while True:
        Component = []
        while len(Ripe)>0:
            i,j         = Ripe.pop()
            Closed[i,j] = True
            Component.append((i,j))
            for delta_i in deltas:
                for delta_j in deltas:
                    if delta_i==0 and delta_j==0: continue     # ignore case where we don't move
                    i1 = i+delta_i
                    j1 = j+delta_j
                    if i1<0 or i1>=nx or j1<0 or j1>=ny: continue   # Don't move outside image
                    
                    if Image[i1,j1,BLUE]>threshold and not Closed[i1,j1] and not (i1,j1) in Ripe:
                        Ripe.add((i1,j1))
        yield Component
        
        next_set = find_first_ripe()
        if next_set == None: return
        Ripe    = set([next_set])               



def parse_tuple(s):
    return tuple([int(x) for x in s[1:-1].split(',')])

def remove_false_findings(Image,threshold=-1,nx=256,ny=256):
    component_file = join(gettempdir(),f'{uuid4()}.txt')
    
    Areas = []
    
    with open(component_file,'w') as temp:
        for Component in generate8components(Image,threshold=threshold,nx=nx,ny=ny):
            if len(Component)>1:
                Areas.append(len(Component))
                temp.write(' '.join([f'({x},{y})' for x,y in Component])  + '\n')
                
    P = mean(Areas)- std(Areas)
    
    n,bins = histogram(Areas,bins=25)    
    
    Mask = zeros((nx,ny,3)) 
    
    with open(component_file,'r') as temp:
        for line in temp:
            Component = [parse_tuple(s) for s in line.strip().split()]
            if len(Component)>P:
                for i,j in Component:
                    Mask[i,j,BLUE] = 1
                
    remove(component_file)
    
    return Mask,n,bins

def plot_hist(n,bins,axs=None,title=''):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    axs.bar(center, n, align='center', width=width)    
    axs.set_title(title) 
    
def segment(args,image_id):    
    Image    = read_image(path=args.path,image_id=image_id,image_set=args.image_set)
    nx,ny,_  = Image.shape
    
    fig      = figure(figsize=(20,20))
    axs      = fig.subplots(2, 3) 

    im       = axs[0,0].imshow(Image[:,:,BLUE],cmap=cm.get_cmap('Blues'))
    axs[0,0].axes.xaxis.set_ticks([])
    axs[0,0].axes.yaxis.set_ticks([]) 
    axs[0,0].set_title(image_id)
    fig.colorbar(im, ax=axs[0,0], orientation='vertical')
    
    Thresholds,ICVs,n,bins = otsu(Image,nx,ny)
    plot_hist(n,bins,axs=axs[1,1],title='Blue levels')

    axs[1,0].plot(range(len(ICVs)),ICVs, c='r', label='ICV')
    axs[1,0].set_title('Intra-class variance')
    axs[1,0].set_xlabel('Iteration')
    axs[1,0].set_ylabel('ICV')
    ax2t = axs[1,0].twinx()
    ax2t.plot(range(len(Thresholds)),Thresholds,c='b', label='Threshold')
    ax2t.set_ylabel('Threshold')
    axs[1,0].legend(loc='lower center',framealpha=0.5)
    ax2t.legend(loc='center right')
    
    Partitioned   = zeros((nx,ny,3))
    for i in range(nx):
        for j in range(ny):
            if Image[i,j,BLUE]>Thresholds[-1]:
                Partitioned[i,j,BLUE] = Image[i,j,BLUE]
                
    axs[0,1].imshow(Partitioned) 
    axs[0,1].axes.xaxis.set_ticks([])
    axs[0,1].axes.yaxis.set_ticks([]) 
    axs[0,1].set_title('Partitioned')
    
    Mask,n,bins = remove_false_findings(Image,threshold=Thresholds[-1],nx=nx,ny=ny)
    
    plot_hist(n,bins,axs=axs[1,2])
    axs[0,2].imshow(Mask)
    
    fig.suptitle(f'{"+".join([Descriptions[label] for label in Training[image_id]])  }')

    fig.savefig(join(args.figs,f'{image_id}.png'))
    
    if not args.show:
        close(fig)
    
if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Segment HPA data using Otsu\'s algorithm')
    parser.add_argument('--path',                default=r'C:\data\hpa-scc')
    parser.add_argument('--image_set',           default = 'train512x512')
    parser.add_argument('--image_id',            default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0')
    parser.add_argument('--show',                default=False, action='store_true')
    parser.add_argument('--figs',                default= './figs')
    parser.add_argument('--all',                 default=False, action='store_true')
    parser.add_argument('--sample', type=int,    default=0)
    args     = parser.parse_args()
    
    Training = read_training_expectations(path=args.path)
    if args.sample:
        for image_id in sample(list(Training.keys()),args.sample):
            print (f'{image_id}.')
            segment(args,image_id)     
    elif args.all:
        n = len(Training)
        for image_id in sorted(Training.keys()):
            print (f'{image_id}. {n} remaining')
            segment(args,image_id) 
            n -= 1
    else:
        segment(args,args.image_id)

    
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
    if args.show:
        show()