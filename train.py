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
from numpy                      import asarray, stack
from os.path                    import join, exists
from os                         import walk
from shutil                     import copy
from skimage                    import io, transform
from time                       import time
from torch                      import from_numpy,unsqueeze,FloatTensor,save
from torch.nn                   import Module, Conv3d, MaxPool3d, Linear,MSELoss
from torch.nn.functional        import relu
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
        with open(join(path,file_name)) as train:
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
        # print (x.shape)
        x = self.pool(relu(self.conv1(x.float())))
        # print (x.shape)
        x = self.pool(relu(self.conv2(x)))
        # print (x.shape)
        x = x.view(-1, 16 * 126 * 126)
        # print (x.shape)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow1(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    imshow(inp)
    if title is not None:
        title(title)


if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Train with HPA data')
    parser.add_argument('--path',
                        default = r'd:\data\hpa-scc',
                        help    = 'Path to data')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Identified subset of data-- e.g. train512x512')
    parser.add_argument('--freq',
                        default = 10,
                        type    = int)
    parser.add_argument('--momentum',
                        default = 0.9,
                        type    = float)
    parser.add_argument('--lr',
                        default = 0.001,
                        type    = float)
    parser.add_argument('--n',
                        default = 10,
                        type    = int)
    parser.add_argument('--prefix',
                        default = 'log')
    parser.add_argument('--suffix',
                        default = '.csv')
    parser.add_argument('--saveweights',
                        default = 'weights')
    args          = parser.parse_args()

    logs          = get_logfile_names(False,None,args.prefix,args.suffix)
    i             = len(logs)
    logfile_name = f'{args.prefix}{i+1}{args.suffix}'
    while exists(logfile_name):
        i += 1
        logfile_name = f'{args.prefix}{i+1}{args.suffix}'

    training_data   = CellDataset()
    training_loader = DataLoader(training_data,batch_size=7)

    model     = Net()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

    print (model)
    with open(logfile_name,'w') as logfile:
        logfile.write(f'image_set,{args.image_set}\n')
        logfile.write(f'lr,{args.lr}\n')
        logfile.write(f'momentum,{args.momentum}\n')
        for epoch in range(1,args.n+1):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(training_loader, 0):
                inputs, labels = data # get the inputs; data is a list of [inputs, labels]
                optimizer.zero_grad()  # zero the parameter gradients

                # forward + backward + optimize
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % args.freq == 0:
                    print(f'{epoch}, {i}, {running_loss / args.freq}')
                    logfile.write(f'{epoch}, {i}, {running_loss / args.freq}\n')
                    logfile.flush()
                    save_weights_path = f'{args.saveweights}.pth'
                    if exists(save_weights_path):
                        copy(save_weights_path,f'{args.saveweights}.pth.bak')
                    save(model.state_dict(),save_weights_path )
                running_loss = 0.0


    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
