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
from multiprocessing            import cpu_count
from numpy                      import asarray, stack, argmax, mean
from os.path                    import join, exists
from os                         import walk
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
        with open(file_name) as index:
            rows = reader(index)
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

        return unsqueeze(img,1),FloatTensor(classes)

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
        x = softmax(x,dim=1)
        return x

def get_logfile_name(args):
    logs          = get_logfile_names(False,None,args.prefix,args.suffix)
    i             = len(logs)
    logfile_path = join(args.logdir, f'{args.prefix}{i+1}{args.suffix}')
    while exists(logfile_path):
        i += 1
        logfile_path = join(args.logdir, f'{args.prefix}{i+1}{args.suffix}')
    return logfile_path

def train(args):
    training_loader = DataLoader(CellDataset(file_name = 'training.csv' if args.index == None else args.index),
                                 batch_size = cpu_count())

    model     = Net()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(),
                    lr = args.lr,
                    momentum = args.momentum)

    print (model)
    with open(get_logfile_name(args),'w') as logfile:
        logfile.write(f'image_set,{args.image_set}\n')
        logfile.write(f'lr,{args.lr}\n')
        logfile.write(f'momentum,{args.momentum}\n')
        for epoch in range(1,args.n+1):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(training_loader, 0):
                inputs, labels = data # get the inputs; data is a list of [inputs, labels]
                optimizer.zero_grad()  # zero the parameter gradients

                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % args.freq == 0:
                    print(f'{epoch}, {i}, {running_loss / args.freq}')
                    logfile.write(f'{epoch}, {i}, {running_loss / args.freq}\n')
                    logfile.flush()
                    save_weights_path = f'{args.weights}.pth'
                    if exists(save_weights_path):     # backup old weights in case save fails
                        copy(save_weights_path,f'{args.weights}.pth.bak')
                    save(model.state_dict(),save_weights_path )
                    running_loss = 0.0

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

def validate(args):
    model = Net()
    model.load_state_dict(load(f'{args.weights}.pth'))
    model.eval()
    validation_loader = DataLoader(CellDataset(file_name = 'validation.csv'  if args.index == None else args.index),
                                   batch_size = cpu_count())
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
                        help    = 'Controls frequency with which data logged/saved')
    parser.add_argument('--momentum',
                        default = 0.9,
                        type    = float,
                        help    = 'Momentum')
    parser.add_argument('--lr',
                        default = 0.01,
                        type    = float,
                        help    = 'Learning Rate')
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
    args          = parser.parse_args()

    {
        'train'    : train,
        'validate' : validate,
        'test'     : test
     }[args.action](args)

    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
