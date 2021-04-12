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

from argparse            import ArgumentParser
from matplotlib.pyplot   import figure, close, savefig, show
from numpy               import load, mean, argmax,argmin,argsort
from os                  import walk
from os.path             import exists, join
from random              import seed, shuffle
from torch               import unsqueeze, from_numpy, FloatTensor, save, load as reload
from torch.nn            import Module, Conv3d, MaxPool3d, Linear, MSELoss, Dropout
from torch.nn.functional import relu, softmax
from torch.optim         import SGD
from torch.utils.data    import Dataset, DataLoader
from utils               import Logger, Timer

# Allowable actions for program

TRAIN          = 'train'
TEST           = 'test'
VALIDATE       = 'validate'
VISUALIZE_DATA = 'visualize'

# CellDataset
#
# Read a slice of data that was packed up by slice.py, and allow
# a neural network to train on it.

class CellDataset(Dataset):
    def __init__(self,
                 base_name      = 'train',
                 path           = './',
                 seq            = 1,
                 shuffle_images = False):
        npzfile      = load(join(path,f'{base_name}{seq}.npz'),allow_pickle=True)
        self.Images  = npzfile['Images']
        self.Targets = npzfile['Targets']
        self.Indices = list(range(self.Images.shape[0]))
        if shuffle_images:
            shuffle(self.Indices)

    def __len__(self):
        return self.Images.shape[0]

    def __getitem__(self, idx):
        index  = self.Indices[idx]
        Inputs = unsqueeze(from_numpy(self.Images[index]),1)
        Labels = FloatTensor([1 if i in self.Targets[index] else 0 for i in range(18)])
        return Inputs, Labels

def visualize(data,
              path = './',
              seq  = 1):
    dataset   = CellDataset(data,path=path,seq=1)
    Images    = dataset.Images
    N,K,nx,ny   = Images.shape
    print (f'{path} {data}, nx={nx}, ny={ny}')
    for i in range(N):
        fig    = figure(figsize=(10,10))
        axs    = fig.subplots(2,2)
        for k in range(K):
            axs[k//2,k%2].imshow(Images[i,k,:,:],cmap='gray')
        savefig(join(path,f'{data}{seq}-{i}'))
        close(fig)

if __name__=='__main__':
    parser = ArgumentParser('Train with HPA data')
    parser.add_argument('action',
                        choices = [VISUALIZE_DATA, TRAIN, TEST, VALIDATE],
                        help    = 'Action: train network, validate angainst held out data, or predict from test dataset')
    parser.add_argument('data')
    parser.add_argument('--path',
                        default = 'data',
                        help    = 'Folder for data files')

    args          = parser.parse_args()
    if args.action == VISUALIZE_DATA:
        visualize(args.data,path=args.path)
    else:
        print (f'{args.action} not implemented')
