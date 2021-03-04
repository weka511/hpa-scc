from argparse         import ArgumentParser
from matplotlib.image import imread
from numpy            import zeros, int8, amax, load, savez
from os               import getpid
from os.path          import join
from psutil           import Process
from time             import time
from visualize        import read_descriptions, read_training_expectations

RED         = 0
GREEN       = 1
BLUE        = 2
YELLOW      = 3
NCOLOURS    = 3

colours     = ['red',
               'green',
               'blue',
               'yellow'
              ]

meanings    = ['Microtubules',
               'Protein/antibody',
               'Nuclei channels',
               'Endoplasmic reticulum channels'
              ]

class MemoryReporter:
    def __init__(self):
        self.process = Process(getpid())
        self.initial = self.process.memory_full_info()
        print (f'Process {self.process.pid}, starting with Unique Set Size = {self.initial.rss:,} bytes')
    def check(self):
        full_info = self.process.memory_full_info()
        print (f'{full_info.rss:,} diff= {full_info.rss-self.initial.rss:,}')

if __name__=='__main__':
    start    = time()
    reporter = MemoryReporter()
    parser   = ArgumentParser('Visualize HPA data')
    parser.add_argument('--path',      default = r'd:\data\hpa-scc')
    parser.add_argument('--image_set', default = 'train512x512')
    parser.add_argument('--image_id',  default = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0')

    args         = parser.parse_args()

    Descriptions = read_descriptions('descriptions.csv')
    Training     = read_training_expectations(path=args.path)
    Singletons   = [(image_id, classes) for image_id,classes in Training.items() if len(classes)==1]
    stride       = 2
    N            = 4096
    mx           = 256
    my           = 256
    Images       = zeros((mx,my,4,N),dtype=int8)
    print (Images.shape, Images.shape[0]*Images.shape[1]*Images.shape[2]*Images.shape[3])
    Targets = []
    for k in range(N):
        image_id,classes = Singletons[k]
        for column,colour in enumerate([BLUE,RED,YELLOW,GREEN]):
            file_name        = f'{image_id}_{colours[colour]}.png'
            path_name        = join(args.path,args.image_set,file_name)
            grey_scale_image = imread(path_name)
            nx,ny            = grey_scale_image.shape
            mm = amax(grey_scale_image)

            for i in range(mx):
                for j in range(my):
                    Images[i,j,colour,k] = int8(128*grey_scale_image[2*i,2*j]/mm)
        Targets.append(classes)

    reporter.check()
    print (Images.shape, Images.shape[0]*Images.shape[1]*Images.shape[2]*Images.shape[3])

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    start = time()
    savez('test.npz',Images=Images,Targets=Targets)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')

    start = time()
    npzfile = load('test.npz')

    print(npzfile.files)
    print (npzfile['Images'].shape)
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
