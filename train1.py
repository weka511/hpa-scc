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

from argparse                   import ArgumentParser
from segment2                   import read_training_expectations
from segment                    import read_image
from time                       import time
from torch.utils.data           import Dataset
from torch.nn                   import Module, Conv2d, MaxPool2d, Linear,CrossEntropyLoss
from torch.nn.functional        import relu
from torch.optim                import SGD
from torch                      import from_numpy




class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=4, out_channels=6, kernel_size=5,stride=1,padding=1)
        self.pool  = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1   = Linear(16 * 5 * 5, 120)
        self.fc2   = Linear(120, 84)
        self.fc3   = Linear(84, 10)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__=='__main__':
    start  = time()
    parser = ArgumentParser('Train with HPA data')
    parser.add_argument('--path',
                        default = r'd:\data\hpa-scc',
                        help    = 'Path to data')
    parser.add_argument('--image_set',
                        default = 'train512x512',
                        help    = 'Identified subset of data-- e.g. train512x512')
    parser.add_argument('--momentum', default = 0.9,   type=int)
    parser.add_argument('--lr',       default = 0.001, type=float)
    args   = parser.parse_args()


    # training_set   = CellData()
    # dataiter       = iter(training_set)

    net            = Net()
    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(),
                          lr       = args.lr,
                          momentum = args.momentum)
    image_id = '5c27f04c-bb99-11e8-b2b9-ac1f6b6435d0'
    image    = read_image(path=args.path,image_set=args.image_set,image_id=image_id)
    net(from_numpy(image))
    elapsed = time() - start
    minutes = int(elapsed/60)
    seconds = elapsed - 60*minutes
    print (f'Elapsed Time {minutes} m {seconds:.2f} s')
