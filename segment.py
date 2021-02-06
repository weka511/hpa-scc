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
from   csv               import reader
from   math              import isqrt
from   matplotlib.pyplot import figure, show, cm, close,scatter
from   matplotlib.image  import imread
from   numpy             import zeros, array, var, mean, std, histogram
from   os                import remove
from   os.path           import join,basename,exists
from   random            import sample
from   sys               import float_info, exc_info
from   tempfile          import gettempdir
from   time              import time
from   uuid              import uuid4
from   traceback         import print_exc

RED                = 0      # Channel number for Microtubules 
GREEN              = 1      # Channel number for Protein of interest
BLUE               = 2      # Channel number for Nucelus
YELLOW             = 3      # Channel number for Endoplasmic reticulum
NCHANNELS          = 4      # Number of channels
NRGB               = 3      # Number of actual channels for graphcs

COLOUR_NAMES       = ['red',
                      'green',
                      'blue', 
                      'yellow']

IMAGE_LEVEL_LABELS = ['Microtubules', 
                      'Protein/antibody',
                      'Nuclei channels',
                      'Endoplasmic reticulum channels']

DESCRIPTIONS       = [
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




# read_training_expectations
# 
# Read and parse the  training image-level labels 
#
# Parameters:
#     path       Path to image-level labels 
#     file_name  Name of image-level labels file

def read_training_expectations(path=r'C:\data\hpa-scc',file_name='train.csv'):
    with open(join(path,file_name)) as train:
        rows = reader(train)
        next(rows)
        return {row[0]: [int(label) for label in row[1].split('|')] for row in rows}


# read_image
#
# Read set of images (keeping Yellow separate)
#
# Parameters:
#    path            Location of data
#    image_set       Identifies high or low res
#    image_id        Identified image

def read_image(path        = r'C:\data\hpa-scc',
               image_set   = 'train512512',
               image_id    = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'):
    Image = None
    for channel in range(NCHANNELS):
        image_mono = imread(join(path,
                                 image_set,
                                 f'{image_id}_{COLOUR_NAMES[channel]}.png'))
        if channel==0:
            nx,ny          = image_mono.shape
            Image          = zeros((nx,ny,NCHANNELS))
        Image[:,:,channel] = image_mono
    return Image        

# create_selection

def create_selection(Image,
                 Selector = [
                     [1,0,0,1],
                     [0,1,0,1],
                     [0,0,1,0]]):
    nx,ny,_ = Image.shape
    Product = zeros((nx,ny,NRGB))
    Matrix  = array(Selector)
    for i in range(nx):
        for j in range(ny):
            for k in range(NRGB):
                Product[i,j,k] = sum([Matrix[k,l] * Image[i,j,l] for l in range(NCHANNELS)])
  
    return Product

# plot_hist
#
# Plot a histogram from numpy
#
# Parameters:
#     n
#     bins
#     axs
#     title

def plot_hist(n,bins,axs=None,title=None,channel=BLUE): 
    axs.bar((bins[:-1] + bins[1:]) / 2,
            n,
            align = 'center',
            width = 0.7 * (bins[1] - bins[0]),
            color = COLOUR_NAMES[channel][0])
    axs.axes.xaxis.set_ticks([])
    axs.axes.yaxis.set_ticks([])
    if title!= None:
        axs.set_title(title) 

# generate_neighbours
#
# Used to iterate through neighbours of a point
#
# Parameters:
#     x
#     y
#     delta

def generate_neighbours(x,y,delta=[-1,0,1]):
    for dx in delta:
        for dy in delta:
            if dx==0 and dy==0: continue
            yield x + dx, y + dy

# parse_tuple
# 
# Parse a character representation of a tuple '(a,b)' into an actual tuple (a,b)
# 
def parse_tuple(s):
    return tuple([int(x) for x in s[1:-1].split(',')])


class CMask:
    def __init__(self,nx,ny,n_channels=NCHANNELS):
        self.nx         = nx
        self.ny         = ny
        self.n_channels = n_channels
        self.Mask       = zeros((nx,ny,n_channels))
        
    def set(self,x,y,channel,value=1):
        if channel ==YELLOW:
            self.Mask[x,y,RED] = value
            self.Mask[x,y,GREEN] = value
        else:
            self.Mask[x,y,channel] = value
            
    def establish_background(self,background=0.5):
        for i in range(self.nx):
            for j in range(self.nx):
                if all(self.Mask[i,j,k]==0 for k in range(self.n_channels)):
                    for k in range(3):
                            self.Mask[i,j,k] = background
     
    def show(self,ax):
        ax.imshow(self.Mask[:,:,0:-1])

            
# otsu
#
# Segment image using Otsu's method
#
# Parameters:
#     Image
#     nx
#     ny
#     tolerance
#     N
#     channel

def otsu(Image,nx=256,ny=256,tolerance=0.0001,N=50,channel=BLUE):
    def get_icv(threshold):
        P1   = [intensity for intensity in Intensities if intensity<threshold]
        P2   = [intensity for intensity in Intensities if intensity>threshold]        
        var1 = var(P1)
        var2 = var(P2)
        return (len(P1)*var1 + len(P2)*var2)/(len(P1) + len(P2))  
    
    Intensities = [Image[i,j,channel] for i in range(nx) for j in range(ny)]
    n, bins = histogram(Intensities)

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


# generate8components

def generate8components(Image,threshold=0.5,nx=512,ny=512,deltas=[-1,0,1],channel=BLUE):
    def find_first_ripe():
        for i in range(nx):
            for j in range(ny):
                if Open[i,j] and not Closed[i,j]:
                    return i,j
    
    Open      = zeros((nx,ny),dtype=bool) # Set of points that are potentially in a component
    
    for i in range(nx):
        for j in range(ny):
            if Image[i,j,channel]>threshold:
                Open[i,j] = True
                
    Closed    = zeros((nx,ny),dtype=bool)  # Set of points that have been processed already
    
    Ripe = set([find_first_ripe()])        # Here is where we start building a component
    while True:
        Component = []
        while len(Ripe)>0:
            i,j         = Ripe.pop()
            Closed[i,j] = True
            Component.append((i,j))
            for i1,j1 in generate_neighbours(i,j):
                if i1<0 or i1>=nx or j1<0 or j1>=ny: continue   # Don't move outside image       
                if Image[i1,j1,channel]>threshold and not Closed[i1,j1] and not (i1,j1) in Ripe:
                    Ripe.add((i1,j1))
        yield Component
        
        next_set = find_first_ripe()
        if next_set == None: return
        Ripe    = set([next_set])               




# generate_components
#
# Generte components whose area exceeds a specified minimum 
# (using len as a proxy for area)
#
# Parameters:
#      component_file
#      P               Minimum area

def generate_components(component_file,P=0):
    with open(component_file,'r') as temp:
        for line in temp:
            Component = [parse_tuple(s) for s in line.strip().split()]
            if len(Component)>P:
                yield Component
                
# remove_false_findings
#
# Paramaters
#      Image
#      threshold      = -1,
#      nx             = 256,
#      ny             = 256,
#      nsigma         = 1.0,
#      channel        = BLUE,
#      component_file 

def remove_false_findings(Image,
                          threshold      = -1,
                          nx             = 256,
                          ny             = 256,
                          nsigma         = 1.0,
                          channel        = BLUE,
                          component_file = join(gettempdir(),f'{uuid4()}.txt')):
    
    
    Areas = []
    
    with open(component_file,'w') as temp:
        for Component in generate8components(Image,threshold=threshold,nx=nx,ny=ny,channel=channel):
            if len(Component)>1:
                Areas.append(len(Component))
                temp.write(' '.join([f'({x},{y})' for x,y in Component])  + '\n')
                
    P      = mean(Areas) - nsigma*std(Areas)
    n,bins = histogram(Areas,bins=25)    
    
    Mask    = CMask(nx,ny)
    for Component in generate_components(component_file,P=P):
        for i,j in Component:
            Mask.set(i,j,channel)
    
    return Mask,n,bins,P


        
# segment_channel

def segment_channel(Image,
                    channel        = BLUE,
                    component_file = join(gettempdir(),f'{uuid4()}.txt')):    
    
    nx,ny,_  = Image.shape
    
    Thresholds,ICVs,n,bins = otsu(Image,nx,ny,channel=channel)
    
    Mask,n1,bins1,P = remove_false_findings(
                          Image,
                          threshold      = Thresholds[-1],
                          nx             = nx,
                          ny             = ny,
                          channel        = channel,
                          component_file = component_file)
        
    return P,Thresholds,ICVs,n,bins,n1,bins1,Mask

# display_channel

def display_channel(Image, image_id,
                    Thresholds = [],
                    ICVs       = [],
                    n          = [],
                    bins       = [],
                    n1         = [],
                    bins1      = [],
                    Mask       = None,
                    channel    = BLUE,
                    cmap       = 'Blues',
                    path       = './',
                    show       = False):
    nx,ny,_  = Image.shape
    
    fig      = figure(figsize=(20,20))
    axs      = fig.subplots(2, 3) 

    im       = axs[0,0].imshow(Image[:,:,channel],cmap=cm.get_cmap(cmap))
    axs[0,0].axes.xaxis.set_ticks([])
    axs[0,0].axes.yaxis.set_ticks([]) 
    axs[0,0].set_title(image_id)
    fig.colorbar(im, ax=axs[0,0], orientation='vertical')
    
    plot_hist(n,bins,axs=axs[1,1],title=IMAGE_LEVEL_LABELS[channel],channel=channel)

    axs[1,0].plot(range(len(ICVs)),ICVs, c='r', label='ICV')
    axs[1,0].set_title('Intra-class variance')
    axs[1,0].set_xlabel('Iteration')
    axs[1,0].set_ylabel('ICV')
    ax2t = axs[1,0].twinx()
    ax2t.plot(range(len(Thresholds)),Thresholds,c='b', label='Threshold')
    ax2t.set_ylabel('Threshold')
    axs[1,0].legend(loc='lower center',framealpha=0.5)
    ax2t.legend(loc='center right') 
    
    Partitioned   = zeros((nx,ny,NRGB))
    for i in range(nx):
        for j in range(ny):
            if Image[i,j,channel]>Thresholds[-1]:
                if channel<YELLOW:
                    Partitioned[i,j,channel] = Image[i,j,channel]
                else:
                    Partitioned[i,j,RED] = Image[i,j,channel]
                    Partitioned[i,j,GREEN] = Image[i,j,channel]

    axs[0,1].imshow(Partitioned) 
    axs[0,1].axes.xaxis.set_ticks([])
    axs[0,1].axes.yaxis.set_ticks([]) 
    axs[0,1].set_title('Partitioned')    
    plot_hist(n1,bins1,axs=axs[1,2],channel=channel,title='Components')
 
    Mask.establish_background()
    Mask.show( axs[0,2])
    
    fig.suptitle(f'{"+".join([DESCRIPTIONS[label] for label in Training[image_id]])  }')

    fig.savefig(join(path,f'{image_id}_{COLOUR_NAMES[channel]}.png'))
    
    if not show:
        close(fig)


            
# get_thinned
#
# Eliminate interior points from compoent

def get_thinned(Component,n=4):
    # is_isolated
    #
    # Establish whether point is isolated by counting neighbours.
    
    def is_isolated(x,y):
        count = 0
        for x1,y1 in generate_neighbours(x,y,delta=[-1,0,1]):
            if (x1,y1) in Neighbours:
                count+=1
                if count>n:
                    return False
        return True
    Neighbours = set(Component)
    return [(x,y) for (x,y) in Component if is_isolated(x,y)]


        
def display_thinned(Thinned,Image, background = 0.5):
    fig        = figure(figsize=(10,10))
    axs        = fig.subplots(2, 2)
    index      = 0
    nx,ny,_    = Image.shape

    for channel in [BLUE, RED, GREEN, YELLOW]:
        Mask = CMask(nx,ny)
        
        for  point in Thinned[0]:
            for (x,y) in point:
                Mask.set(x,y,BLUE)
        
        if channel!=BLUE:
            for  point in Thinned[index]:
                for (x,y) in point:
                    Mask.set(x,y,channel)
          
        Mask.establish_background()
        Mask.show(axs[index//2][index%2])
        index += 1


# segment
#
# Segment all channels for one image. This is the supervisor that coordinates  functions that
# perform the actual detailed work
#
# Parameters:
#     Image       The image to ge segmented
#     image_id    Image Identifier for use in headings
#     path        Location of plot files
#     show        Indicates whter images are to be displayed (or only saved)
#     channels    The channels to be segments
#     cmaps       Colour maps for displaying each channel

def segment(Image, image_id, 
            path     = './', 
            show     = False,
            channels = [BLUE, RED, GREEN, YELLOW],
            cmaps    = {BLUE:'Blues', RED:'Reds', GREEN:'Greens', YELLOW:'YlOrBr'},
            keep_temp = False):
    component_files = []
    Thinned         = []
    try:
        for channel in channels:
            component_files.append(join(gettempdir(),f'{uuid4()}.txt'))
            P,Thresholds,ICVs,n,bins,n1,bins1,Mask = segment_channel(
                                                            Image, 
                                                            channel        = channel, 
                                                            component_file = component_files[-1])
            display_channel(Image, image_id, 
                            Thresholds     = Thresholds,
                            ICVs           = ICVs,
                            n              = n,
                            bins           = bins,
                            n1             = n1,
                            bins1          = bins1,
                            Mask           = Mask,
                            channel        = channel, 
                            cmap           = cmaps[channel],
                            path           = path,
                            show           = show)
            
            Thinned.append([get_thinned(Component,n=5) for Component  in generate_components(component_files[-1],P=P)])
            
        display_thinned(Thinned,Image)
            
    except Exception as _:
        print (f'{image_id} {exc_info()[0]}')
        print_exc() 
    finally:
        if keep_temp: return
        for component_file in component_files:
            if exists(component_file):  # If there was an exception, file might not actually exist!
                remove(component_file)
        


if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Segment HPA data using Otsu\'s algorithm')
    parser.add_argument('--path',             default=r'C:\data\hpa-scc')
    parser.add_argument('--image_set',        default = 'train512x512')
    parser.add_argument('--image_id',         default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0')
    parser.add_argument('--show',             default=False, action='store_true')
    parser.add_argument('--figs',             default= './figs')
    parser.add_argument('--all',              default=False, action='store_true')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--keep_temp',        default=False, action='store_true')
    args     = parser.parse_args()
    
    Training = read_training_expectations(path=args.path)
    if args.sample:
        n = args.sample
        for image_id in sample(list(Training.keys()),args.sample):
            print (f'{image_id}. {n} remaining')
            segment(read_image(path      = args.path,
                               image_id  = image_id,
                               image_set = args.image_set),
                    image_id = image_id,
                    path     = args.figs,
                    show     = args.show)
            n-=1
    elif args.all:
        n = len(Training)
        for image_id in sorted(Training.keys()):
            print (f'{image_id}. {n} remaining')
            segment(read_image(path      = args.path,
                               image_id  = image_id,
                               image_set = args.image_set),
                    image_id = image_id,
                    path     = args.figs,
                    show     = args.show)
            n -= 1
    else:
        segment(read_image(path      = args.path,
                           image_id  = args.image_id,
                           image_set = args.image_set),
                image_id  = args.image_id,
                path      = args.figs,
                show      = args.show,
                keep_temp = args.keep_temp)
  
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    
    if args.show:
        show()
        
# obsolete

# combine

def anchor_components(Image,image_id, channels=[BLUE,RED],figs='./', show=False, component_files=[],P=[],nx=512,ny=512): #FIXME
    def get_centroid(component):
        sum_x = 0
        sum_y = 0
        for x,y in component:
            sum_x += x
            sum_y += y
        return (sum_x/len(component),sum_y/len(component))
    
    def get_nearest_centroid(x,y,Centroids):
        min_distance = float_info.max
        index        = None
        for i in range(len(Centroids)):
            x0,y0    = Centroids[i]
            distance = (x0-x)**2 + (y0-y)**2
            if distance<min_distance:
                index         = i
                min_distance = distance
        return index
    
    Base      = []
    Centroids = []
    for component in generate_components(component_files[channels[0]],P=P[channels[0]]):
        Base.append(component)
        Centroids.append(get_centroid(component))
        
    Assignments=[]
    for component in generate_components(component_files[channels[1]],P=P[channels[1]]):
        x,y   = get_centroid(component)
        index = get_nearest_centroid(x,y,Centroids)
        Assignments.append(index)
        x0,y0 = Centroids[index]
        print (x,y,x0,y0)
    nrows    = isqrt(len(Centroids))
    ncols    = len(Centroids)//nrows + len(Centroids)%nrows
    fig      = figure(figsize=(20,20))
    axs      = fig.subplots(nrows,ncols)
    i        = 0
    for base in generate_components(component_files[channels[0]],P=P[channels[0]]):
        Matrix = zeros((nx,ny,NRGB))
        for x,y in base:
            Matrix[x,y,channels[0]] = 1
        j = 0
        for component in generate_components(component_files[channels[1]],P=P[channels[1]]):
            if Assignments[j]==i:
                for x,y in component:
                    Matrix[x,y,channels[1]] = 1                
            j+=1
        row = i//ncols
        col =  i%ncols
        axs[row,col].imshow(Matrix)
        i+= 1