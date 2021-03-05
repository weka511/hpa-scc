from numpy                      import load
from torch                      import unsqueeze, from_numpy, FloatTensor
from torch.nn                   import Module, Conv3d, MaxPool3d, Linear, MSELoss
from torch.nn.functional        import relu, softmax
from torch.optim                import SGD
from torch.utils.data           import Dataset, DataLoader

class CellDataset(Dataset):
    def __init__(self, file_name        = 'test1.npz'):
        npzfile = load(file_name)
        self.Images  = npzfile['Images']
        self.Targets = npzfile['Targets']
        print (self.Images.shape)
    def __len__(self):
        return self.Images.shape[3]

    def __getitem__(self, idx):
        return unsqueeze(from_numpy(self.Images[idx]),1), FloatTensor([1 if i in self.Targets[idx] else 0 for i in range(18)])

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

if __name__=='__main__':
    dataset = CellDataset()
    loader   = DataLoader(dataset, batch_size=7)

    model     = Net()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9)

    for epoch in range(2):
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
