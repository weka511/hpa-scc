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
from matplotlib.pyplot          import figure, imread, imshow, axis, savefig, close, show, title,suptitle
from numpy                      import array, clip, asarray
from os.path                    import join
from segment2                   import read_training_expectations
from skimage                    import io, transform
from time                       import time
from torch                      import from_numpy,unsqueeze,FloatTensor
from torch.nn                   import Module, Conv2d, MaxPool2d, Linear,MSELoss
from torch.nn.functional        import relu
from torch.optim                import SGD
from torch.utils.data           import Dataset, DataLoader
from torchvision                import transforms, utils

class CellDataset(Dataset):
    def __init__(self,
                 path         = r'd:\data\hpa-scc',
                 image_set    = 'train512x512',
                 file_name    = 'train.csv',
                 transform    =  None):
        self.expectations = read_training_expectations(path=path,file_name=file_name)
        self.image_ids    = list(self.expectations.keys())
        self.transform    = transform
        self.path         = path
        self.image_set    = image_set

    def __len__(self):
        return len(self.expectations)

    def __getitem__(self, idx):
        image_id        = self.image_ids[idx]
        image_name      = f'{image_id}_red.png'
        full_image_name = join(self.path,self.image_set,image_name)
        img             = io.imread(full_image_name)

        if self.transform:
            sample = self.transform(img)
        img = from_numpy(asarray(img))

        classes = [1 if i in self.expectations[image_id] else 0 for i in range(18)]

        return unsqueeze(img,0),FloatTensor(classes)

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels  = 1,
                            out_channels = 6,
                            kernel_size  = 5,
                            stride       = 1,
                            padding      = 1)
        self.pool  = MaxPool2d(2, 2)
        self.conv2 = Conv2d(in_channels  = 6,
                            out_channels = 16,
                            kernel_size  = 5,
                            stride       = 1,
                            padding      = 1)
        self.fc1   = Linear(16 * 126 * 126, 120)
        self.fc2   = Linear(120, 84)
        self.fc3   = Linear(84, 18)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x.float())))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 126 * 126)
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
    parser.add_argument('--freq',     default = 10,  type=int)
    parser.add_argument('--momentum', default = 0.9,   type=float)
    parser.add_argument('--lr',       default = 0.001, type=float)
    parser.add_argument('--n',        default = 10,     type=int)
    args   = parser.parse_args()

    training_data   = CellDataset()
    training_loader = DataLoader(training_data,batch_size=4)
    # Get a batch of training data
    # inputs, classes = next(iter(training_loader))

    # Make a grid from batch
    # out = utils.make_grid(inputs)

    # imshow1(out)

    model     = Net()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(),
                    lr       = args.lr,
                    momentum = args.momentum)

    losses    = []
    print (model)
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
            if (i +1) % args.freq == 0:
                print(f'{epoch + 1}, {i + 1}, {running_loss / args.freq}')
            losses.append(running_loss/args.freq)
            running_loss = 0.0


    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
    show()
