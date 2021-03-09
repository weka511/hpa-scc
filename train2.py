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
from numpy               import load, mean, argmax,argmin
from os                  import walk
from os.path             import exists, join
from random              import sample, seed
from torch               import unsqueeze, from_numpy, FloatTensor, save, load as reload
from torch.nn            import Module, Conv3d, MaxPool3d, Linear, MSELoss
from torch.nn.functional import relu, softmax
from torch.optim         import SGD
from torch.utils.data    import Dataset, DataLoader
from utils               import Logger, Timer

# Allowable actions for program

TRAIN    = 'train'
TEST     = 'test'
VALIDATE = 'validate'

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


# train_epoch
#
# Train for one epoch. Iterates through all slices of data

def train_epoch(epoch,args,model,criterion,optimizer,logger=None):
    m  = 1
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
                    logger.log(f'{epoch}, {m},  {i}, {mean_loss}')
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

# get_predictions
#
# Use model to predict class of input
def get_predictions(model,inputs):             #FIXME - need to handle multiplets
    def get_prediction(probabilities):
        prediction = argmax(probabilities)
        return prediction, probabilities[prediction]

    output      = model(inputs)
    predictions = [get_prediction(output[i].detach().numpy()) for i in range(len(output))]
    return [a for a,_ in predictions],[b for _,b in predictions]

def get_threshold(Probabilities,Labels,nsteps=25):
    def match(call,label):
        return 0 if call==label else 1
    Scores = []
    for i in range(nsteps+1):
        threshold =i/nsteps
        score = 0
        for probabilities,labels in zip(Probabilities,Labels):
            calls = [1 if p>threshold else 0 for p in probabilities]
            score += sum([match(call,label) for call,label in zip(calls,labels)])
        Scores.append(score)
    index = argmin(Scores)
    j     = index
    while Scores[j+1]==Scores[index]:
        j+=1
    return 0.5* (index+j)/nsteps

def get_test_score(threshold,Probabilities,Labels):
    def match(call,label):
        return 0 if call==label else 1

    score = 0
    for probabilities,labels in zip(Probabilities,Labels):
        calls = [1 if p>threshold else 0 for p in probabilities]
        score += sum([match(call,label) for call,label in zip(calls,labels)])
    return score


def validate(data, checkpoint='chk', batch=8,frequency=10,n1=3,n2=3):
    model       = Net()
    Checkpoints = sorted([join(path,name) for path,_,names in walk('./') for name in names if name.startswith(checkpoint)])
    checkpoint  = reload(Checkpoints[-1])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    dataset                  = CellDataset(file_name =  data)               #FIXME shuffle
    loader                   = DataLoader(dataset, batch_size = batch)
    Validation_Probabilities = []
    Validation_Labels        = []
    threshold                = None
    Test_Probabilities       = []
    Test_Labels              = []

    for i, data in enumerate(loader, 0):
        if i>=n1+n2: break
        inputs, labels = data
        outputs        = model(inputs)
        if i<n1:
            for output in outputs:
                Validation_Probabilities.append(output.detach().numpy())
            for label in labels:
                Validation_Labels.append(label.detach().numpy())
        else:
            for output in outputs:
                Test_Probabilities.append(output.detach().numpy())
            for label in labels:
                Test_Labels.append(label.detach().numpy())

    threshold = get_threshold(Validation_Probabilities,Validation_Labels)
    get_test_score(threshold,Test_Probabilities,Test_Labels)
    # assert(len(Probabilities)==n1*batch),f'{len(Probabilities)},{n1*batch}'
    # assert(len(Labels)==n1*batch),f'{len(Labels)},{n1*batch}'


        # predictions,confidences = get_predictions(model,inputs)

    # matches           = 0
    # mismatches        = 0
    # confidence        = 0.0
    # for i, data in enumerate(loader, 0):
        # inputs, labels = data
        # predictions,confidences = get_predictions(model,inputs)
        # confidence             += sum(confidences)
        # for predicted,expected in zip(predictions,[argmax(labels[i].detach().numpy()) for i in range(len(labels))]):
            # if predicted==expected:
                # matches+=1
            # else:
                # mismatches+=1
        # if (matches+mismatches)%frequency==0:
            # print (f'Confidence = {100*confidence/(matches+mismatches):.1f}%, accuracy={100*matches/(matches+mismatches):.1f}%')

if __name__=='__main__':
    parser = ArgumentParser('Train with HPA data')
    parser.add_argument('action',
                        choices = [TRAIN, TEST, VALIDATE],
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
    seed(42)                                 #FIXME
    model         = Net()
    criterion     = MSELoss()
    optimizer     = SGD(model.parameters(),
                        lr       = args.lr,
                        momentum = args.momentum)

    if args.action == TRAIN:
        with Timer('training network'), \
                    Logger(prefix = args.prefix,
                           suffix = args.suffix,
                           logdir = args.logdir) as logger:
            logger.log(f'lr,{args.lr}')
            logger.log(f'momentum,{args.momentum}')
            epoch0 = restart(args,model,criterion,optimizer) if args.restart!=None else 1
            for epoch in range(epoch0,epoch0+args.n):
                train_epoch(epoch,args,model,criterion,optimizer,logger=logger)
                save({
                        'epoch'                : epoch,
                        'model_state_dict'     : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict()#,
                        # 'loss'                 : loss
                    },
                    f'{args.checkpoint}{epoch}.pth')
    elif args.action == VALIDATE:
        validate(args.data,
                 checkpoint = args.checkpoint,
                 batch      = args.batch,
                 frequency  = args.frequency)
    else:
        print ('Not implemeneted')
