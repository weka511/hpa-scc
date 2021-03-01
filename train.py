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

# https://leonardoaraujosantos.gitbook.io/artificial-inteligence/appendix/pytorch/dataloader-and-datasets

from argparse                   import ArgumentParser
from csv                        import reader
from logs                       import get_logfile_names
from numpy                      import asarray, stack, argmax, mean
from os.path                    import join, exists
from os                         import walk
from random                     import seed, shuffle
from shutil                     import copy
from skimage                    import io, transform
from sys                        import exit
from time                       import time
from torch                      import from_numpy,unsqueeze,FloatTensor,save,load
from torch.nn                   import Module, Conv3d, MaxPool3d, Linear, MSELoss
from torch.nn.functional        import relu, softmax
from torch.optim                import SGD
from torch.utils.data           import Dataset, DataLoader
from torchvision                import transforms, utils

class CellDataset(Dataset):
    def __init__(self,
                 path             = r'd:\data\hpa-scc',
                 image_set        = 'train512x512',
                 file_name        = 'train.csv',
                 transform        =  None,
                 allow_multiplets =  False,
                 allow_negatives  =  False):
        self.expectations = {}
        with open(file_name) as train:
            rows = reader(train)
            next(rows)
            for row in rows:
                class_ids = list(set([int(label) for label in row[1].split('|')]))
                if allow_multiplets or len(class_ids)==1:
                    if allow_negatives or 18 not in class_ids:
                        self.expectations[row[0]] = class_ids
        self.image_ids    = list(self.expectations.keys())
        self.transform    = transform
        self.path         = path
        self.image_set    = image_set

    def __len__(self):
        return len(self.expectations)

    def __getitem__(self, idx):
        image_id        = self.image_ids[idx]
        imgs            = []
        for colour in ['red','green','blue','yellow']:
            image_name      = f'{image_id}_{colour}.png'
            full_image_name = join(self.path,self.image_set,image_name)
            img             = io.imread(full_image_name)

            if self.transform:
                img = self.transform(img)
            imgs.append(asarray(img))

        img = from_numpy(stack(imgs))

        classes = [1 if i in self.expectations[image_id] else 0 for i in range(18)]

        # return img,FloatTensor(classes)
        return unsqueeze(img,1),FloatTensor(classes)

    def shuffle(self):
        shuffle(self.image_ids)

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
        self.fc1   = Linear(in_features  = 16 * 126 * 126,
                            out_features = 120)
        self.fc2   = Linear(in_features  = 120,
                            out_features = 84)
        self.fc3   = Linear(in_features  = 84,
                            out_features = 18)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x.float())))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 126 * 126)
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

# save_weights
def save_weights(model,weights='weights'):
    save_weights_path = f'{weights}.pth'
    if exists(save_weights_path):
        copy(save_weights_path,f'{weights}.pth.bak')
    save(model.state_dict(),save_weights_path )

# get_index_file_name

def get_index_file_name(index=None, default = 'validation.csv'):
    return default  if index == None else index

#  train

# Train network

def train(args):

    dataset = CellDataset(file_name =  get_index_file_name(index   = args.index,
                                                           default = 'training.csv'))
    loader   = DataLoader(dataset, batch_size=args.batch)

    model     = Net()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

    model = Net()

    epoch0 = 1
    if args.restart!=None:
        print (f'Loading checkpoint: {args.restart}')
        checkpoint = load(args.restart)
        print (f'Loaded checkpoint: {args.restart}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch0 = checkpoint['epoch'] + 1
        loss   = checkpoint['loss']

        model.train()

    print (model)
    with open(get_logfile_path(prefix = args.prefix,
                               suffix = args.suffix,
                               logdir = args.logdir),
              'w') as logfile:
        logfile.write(f'image_set,{args.image_set}\n')
        logfile.write(f'lr,{args.lr}\n')
        logfile.write(f'momentum,{args.momentum}\n')
        running_losses = []
        for epoch in range(epoch0,args.n+epoch0):  # loop over the dataset multiple times
            dataset.shuffle()
            for i, data in enumerate(loader, 0):
                inputs, labels = data # get the inputs; data is a list of [inputs, labels]
                optimizer.zero_grad()  # zero the parameter gradients

                # forward + backward + optimize
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_losses.append(loss.item())
                if i % args.freq == 0:
                    mean_loss = mean(running_losses)
                    print(f'{epoch}, {i}, {mean_loss}')
                    logfile.write(f'{epoch}, {i}, {mean_loss}\n')
                    logfile.flush()
                    running_losses.clear()

            if len(running_losses)>0:
                mean_loss = mean(running_losses)
                print(f'{epoch}, {i}, {mean_loss}')
                logfile.write(f'{epoch}, {i}, {mean_loss}\n')
                logfile.flush()
                running_losses.clear()
            save({
                    'epoch'                : epoch,
                    'model_state_dict'     : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss'                 : loss,
                },
                f'{args.checkpoint}{epoch}.pth')


# get_predictions
#
# Use model to predict class of input
def get_predictions(model,inputs):
    def get_prediction(probabilities):
        prediction = argmax(probabilities)
        return prediction, probabilities[prediction]

    output      = model(inputs)
    predictions = [get_prediction(output[i].detach().numpy()) for i in range(len(output))]
    return [a for a,_ in predictions],[b for _,b in predictions]

# validate

def validate(args):
    model       = Net()
    Checkpoints = sorted([join(path,name) for path,_,names in walk('./') for name in names if name.startswith(args.checkpoint)])
    checkpoint  = load(Checkpoints[-1])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    validation_loader = DataLoader(CellDataset(file_name =  get_index_file_name(index=args.index)),
                                   batch_size=args.batch)
    matches           = 0
    mismatches        = 0
    confidence        = 0.0
    for i, data in enumerate(validation_loader, 0):
        inputs, labels = data
        predictions,confidences = get_predictions(model,inputs)
        confidence             += sum(confidences)
        for predicted,expected in zip(predictions,[argmax(labels[i].detach().numpy()) for i in range(len(labels))]):
            if predicted==expected:
                matches+=1
            else:
                mismatches+=1
        if (matches+mismatches)%args.freq==0:
            print (f'Confidence = {100*confidence/(matches+mismatches):.1f}%, accuracy={100*matches/(matches+mismatches):.1f}%')


def test(args):
    print ('Not implemented')
    exit(1)

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Train with HPA data')
    parser.add_argument('action',
                        choices = ['train','test','validate'],
                        help    = 'Action: train network, validate angainst held out data, or predict from test dataset')
    parser.add_argument('--path',
                        default = r'd:\data\hpa-scc',
                        help    = 'Path to data')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Identified subset of data-- e.g. train512x512')
    parser.add_argument('--freq',
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
    parser.add_argument('--prefix',
                        default = 'log',
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
                        default = 7,
                        type    = int,
                        help    = 'Batch size for training/validating')
    parser.add_argument('--checkpoint',
                        default = 'chk')
    parser.add_argument('--restart',
                        default = None)
    parser.add_argument('--seed',
                        default = None)
    args          = parser.parse_args()

    seed(args.seed)

    Actions = {   'train'    : train,
                  'validate' : validate,
                  'test'     : test
     }

    Actions[args.action](args)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
