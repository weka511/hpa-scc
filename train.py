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
from utils               import Logger, Timer, set_random_seed

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

# Net
#
# Simple convolutional neural network

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1    = Conv3d(in_channels  = 4,
                               out_channels = 6,
                               kernel_size  = (1,5,5),
                               stride       = 1,
                               padding      = 1)
        self.pool     = MaxPool3d(2)
        self.conv2    = Conv3d(in_channels  = 6,
                               out_channels = 16,
                               kernel_size  = (1,5,5),
                               stride       = 1,
                               padding      = 1)
        self.fc1      = Linear(in_features  = 8* 16* 1* 30* 30,
                               out_features = 120)

        self.fc2      = Linear(in_features  = 120,
                               out_features = 84)

        self.fc3      = Linear(in_features  = 84,
                               out_features = 18)

    def forward(self, x):
        # print (x.shape)
        x = self.pool(relu(self.conv1(x.float())))
        # print (x.shape)
        x = self.pool(relu(self.conv2(x)))
        # print (x.shape)
        x = x.view(-1, 8* 16* 1* 30* 30)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return softmax(x,dim=1)


def validate(data,path='./'):
    print ('Not implemented')

def test(data,path='./'):
    print ('Not implemented')

# log_non_default
#
# Log any non-default arguments

def log_non_default(args):
    for key,value in vars(args).items():
        if key in ['action', 'prefix', 'suffix']: continue    #...except for these
        if value != parser.get_default(key):
            logger.log (f'{key}, {value}')

# train_epoch
#
# Train for one epoch. Iterates through all slices of data

def train_epoch(epoch,
                base_name    = None,
                path         = './',
                model        = None,
                criterion    = None,
                optimizer    = None,
                logger       = None,
                K            = 1,
                frequency    = 1,
                shuffle_data = False,
                batch_size   = 8):
    m    = 1
    seqs = []
    while exists(file_name:=join(path,f'{base_name}{m}.npz')):
        seqs.append(m)
        m+= 1

    if shuffle_data:
        shuffle(seqs)
    for i in range(len(seqs)):
        print (f'Epoch {epoch}, file {base_name} {seqs[i]}')
        loader    = DataLoader(CellDataset(base_name = base_name,
                                           path      = path,
                                           seq       = seqs[i]),
                               batch_size=batch_size)
        losses    = []

        for k in range(K):
            for j, data in enumerate(loader, 0):
                optimizer.zero_grad()
                inputs, labels = data
                outputs        = model(inputs)
                loss           = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                if j%frequency==0:
                    mean_loss = mean(losses)
                    losses.clear()
                    logger.log(f'{epoch}, {m},  {j}, {mean_loss}')

# restart
#
# Restart from saved data

def restart(restart,
            chks      = 'chks',
            model     = None,
            criterion = None,
            optimizer = None):
    checkpoint_file_name = join(chks,restart)
    print (f'Loading checkpoint: {checkpoint_file_name}')
    checkpoint = reload(checkpoint_file_name)
    print (f'Loaded checkpoint: {checkpoint_file_name}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    model.train()
    return epoch

if __name__=='__main__':
    parser = ArgumentParser('Train with HPA data')
    parser.add_argument('action',
                        choices = [VISUALIZE_DATA, TRAIN, TEST, VALIDATE],
                        help    = 'Action: train network, validate angainst held out data, or predict from test dataset')
    parser.add_argument('data')
    parser.add_argument('--path',
                        default = 'data',
                        help    = 'Folder for data files')
    parser.add_argument('--prefix',
                        default = 'train',
                        help    = 'Prefix for log file names')
    parser.add_argument('--suffix',
                        default = '.csv',
                        help    = 'Suffix for log file names')
    parser.add_argument('--logdir',
                        default = './logs',
                        help = 'directory for storing logfiles')
    parser.add_argument('--restart',
                        default = None,
                        help   = 'Restart from specified checkpoint')
    parser.add_argument('--n',
                        default = 10,
                        type    = int,
                        help    = 'Number of epochs for training')
    parser.add_argument('--momentum',
                        default = 0.9,
                        type    = float,
                        help    = 'Momentum for optimization')
    parser.add_argument('--lr',
                        default = 0.01,
                        type    = float,
                        help    = 'Learning Rate for optimization')
    parser.add_argument('--K',
                        default = 1,
                        type    = int,
                        help    = 'Number of pasees through one dataset before loading next one')
    parser.add_argument('--frequency',
                        default = 10,
                        type    = int,
                        help    = 'Controls frequency with which data logged')
    parser.add_argument('--batch',
                        default = 8,
                        type    = int,
                        help    = 'Batch size for training/validating')
    parser.add_argument('--chks',
                        default = 'chks',
                        help    = 'Folder for checkpoint files')
    parser.add_argument('--checkpoint',
                        default = 'chk',
                        help    = 'Base of name for checkpoint files')
    parser.add_argument('--seed',
                        default = 42,
                        type    = int,
                        help    = 'Used to seed random number generator')
    args  = parser.parse_args()

    set_random_seed(args.seed)

    if args.action == VISUALIZE_DATA:
        visualize(args.data,path=args.path)
    elif args.action == TRAIN:
        with Timer('training network'), Logger(prefix = args.prefix,
                                               suffix = args.suffix,
                                               logdir = args.logdir) as logger:
            log_non_default(args)
            model         = Net()
            criterion     = MSELoss()
            optimizer     = SGD(model.parameters(),
                                lr       = args.lr,
                                momentum = args.momentum)
            epoch0 = restart(args.restart,
                             chks      = args.chks,
                             model     = model,
                             criterion = criterion,
                             optimizer = optimizer) if args.restart!=None else 1
            for epoch in range(epoch0,epoch0+args.n):
                train_epoch(epoch,
                            base_name  = args.data,
                            path       = args.path,
                            model      = model,
                            criterion  = criterion,
                            optimizer  = optimizer,
                            logger     = logger,
                            K          = args.K,
                            frequency  = args.frequency,
                            batch_size = args.batch)
                save({
                        'epoch'                : epoch,
                        'model_state_dict'     : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict()
                    },
                    join(args.chks, f'{args.checkpoint}{epoch}.pth'))
    elif args.action == TEST:
        test(args.data,path=args.path)
    elif args.action == VALIDATE:
        validate(args.data,path=args.path)
    else:
        print (f'{args.action} not recognized')
