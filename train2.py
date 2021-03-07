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
from logs                import get_logfile_names
from numpy               import load, mean
from os.path             import exists, join
from time                import time
from torch               import unsqueeze, from_numpy, FloatTensor, save, load as reload
from torch.nn            import Module, Conv3d, MaxPool3d, Linear, MSELoss
from torch.nn.functional import relu, softmax
from torch.optim         import SGD
from torch.utils.data    import Dataset, DataLoader

# Timer
#
# This class is used as a wrapper when I want to know the execution time of some code.

class Timer:
    def __init__(self, message = ''):
        self.start   = None
        self.message = message

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time() - self.start
        minutes = int(elapsed/60)
        seconds = elapsed - 60*minutes
        print (f'Elapsed Time {self.message} {minutes} m {seconds:.2f} s')

# CellDataset
#
# Read a slice of data that was packed up by slice.py, and allow
# a neural network to train on it.

class CellDataset(Dataset):
    def __init__(self, file_name = 'dataset1.npz'):
        npzfile = load(file_name,allow_pickle=True)
        self.Images  = npzfile['Images']
        self.Targets = npzfile['Targets']

    def __len__(self):
        return self.Images.shape[0]

    def __getitem__(self, idx):
        return unsqueeze(from_numpy(self.Images[idx]),1), FloatTensor([1 if i in self.Targets[idx] else 0 for i in range(18)])

# Net
#
# Simple connulutional neural network

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv3d(in_channels  = 4,
                            out_channels = 6,
                            kernel_size  = (1,5,5),
                            stride       = 1,
                            padding      = 1)
        self.pool  = MaxPool3d(2)
        self.conv2 = Conv3d(in_channels  = 6,
                            out_channels = 16,
                            kernel_size  = (1,5,5),
                            stride       = 1,
                            padding      = 1)
        self.fc1   = Linear(in_features  = 16 * 62 * 62,
                            out_features = 120)
        self.fc2   = Linear(in_features  = 120,
                            out_features = 84)
        self.fc3   = Linear(in_features  = 84,
                            out_features = 18)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x.float())))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 62 * 62)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return softmax(x,dim=1)

# get_logfile_path

def get_logfile_path(prefix='log',suffix='csv',logdir='./logs'):
    logs          = get_logfile_names(False,None,prefix,suffix)
    i             = len(logs)
    logfile_path = join(logdir, f'{prefix}{i+1}{suffix}')
    while exists(logfile_path):
        i += 1
        logfile_path = join(logdir, f'{prefix}{i+1}{suffix}')
    return logfile_path

# train_epoch
#
# Train for one epoch. Labels through all slices of data
def train_epoch(epoch,args,model,criterion,optimizer):
    m             = 1
    while exists(file_name:=f'{args.data}{m}.npz'):
        print (f'Epoch {epoch}, file {file_name}')
        dataset   = CellDataset(file_name = file_name)
        loader    = DataLoader(dataset, batch_size=args.batch)
        losses    = []

        for k in range(args.K):
            for i, data in enumerate(loader, 0):
                optimizer.zero_grad()
                inputs, labels = data
                outputs        = model(inputs)
                loss           = criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                if i%args.frequency==0:
                    mean_loss = mean(losses)
                    losses.clear()
                    print (f'{epoch}, {m},  {i}, {mean_loss}')
                    logfile.write(f'{epoch}, {m}, {i}, {mean_loss}\n')
                    logfile.flush()
        m+= 1

# restart
#
# Restart from saved data

def restart(args,model,criterion,optimizer):
    print (f'Loading checkpoint: {args.restart}')
    checkpoint = reload(args.restart)
    print (f'Loaded checkpoint: {args.restart}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    # loss  = checkpoint['loss']
    model.train()
    return epoch

if __name__=='__main__':
    parser = ArgumentParser('Train with HPA data')
    parser.add_argument('action',
                        choices = ['train','test','validate'],
                        help    = 'Action: train network, validate angainst held out data, or predict from test dataset')
    parser.add_argument('data')
    parser.add_argument('--frequency',
                        default = 10,
                        type    = int,
                        help    = 'Controls frequency with which data logged')
    parser.add_argument('--momentum',
                        default = 0.9,
                        type    = float,
                        help    = 'Momentum for optimization')
    parser.add_argument('--lr',
                        default = 0.01,
                        type    = float,
                        help    = 'Learning Rate for optimization')
    parser.add_argument('--n',
                        default = 10,
                        type    = int,
                        help    = 'Number of epochs for training')
    parser.add_argument('--K',
                        default = 1,
                        type    = int,
                        help    = 'Number of pasees through one dataset before loading next one')
    parser.add_argument('--prefix',
                        default = 'train',
                        help    = 'Prefix for log file names')
    parser.add_argument('--suffix',
                        default = '.csv',
                        help    = 'Suffix for log file names')
    parser.add_argument('--logdir',
                        default = './logs',
                        help = 'directory for storing logfiles')
    parser.add_argument('--weights',
                        default = 'weights',
                        help    = 'Filename for saving and loading weights')
    parser.add_argument('--index',
                        default = None,
                        help    = 'Name for index file for training/validation')
    parser.add_argument('--batch',
                        default = 8,
                        type    = int,
                        help    = 'Batch size for training/validating')
    parser.add_argument('--checkpoint',
                        default = 'chk',
                        help    = 'Base of name for checkpoint files')
    parser.add_argument('--restart',
                        default = None,
                        help   = 'Restart from specified checkpoint')

    args          = parser.parse_args()
    model         = Net()
    criterion     = MSELoss()
    optimizer     = SGD(model.parameters(),
                        lr       = args.lr,
                        momentum = args.momentum)

    with Timer('(training network)'),  open(get_logfile_path(prefix = args.prefix,
                                                             suffix = args.suffix,
                                                             logdir = args.logdir),
                                            'w') as logfile:
        logfile.write(f'lr,{args.lr}\n')
        logfile.write(f'momentum,{args.momentum}\n')
        epoch0 = restart(args,model,criterion,optimizer) if args.restart!=None else 1
        for epoch in range(epoch0,epoch0+args.n):
            train_epoch(epoch,args,model,criterion,optimizer)
            save({
                    'epoch'                : epoch,
                    'model_state_dict'     : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict()#,
                    # 'loss'                 : loss
                },
                f'{args.checkpoint}{epoch}.pth')
