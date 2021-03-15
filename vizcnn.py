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


from argparse           import ArgumentParser
from matplotlib.pyplot  import figure, axis, imshow, title, show, tight_layout, savefig,suptitle
from numpy              import asarray, multiply, transpose, array, minimum, float32, maximum, mean, std
from os.path            import join
from torch              import load
from torch.nn           import Module, Conv3d, Conv2d
from train2             import Net3C


# plot_filters_single_channel_big
#
# Code snarfed from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
def plot_filters_single_channel_big(t):

    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]


    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)

# plot_filters_single_channel
#
# Code snarfed from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e

def plot_filters_single_channel(t):

    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12

    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()


# plot_filters_multi_channel
#
# Code snarfed from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
def plot_filters_multi_channel(t):
    num_kernels = t.shape[0]
    num_cols = 12
    num_rows = num_kernels
    fig = figure(figsize=(num_cols,num_rows))

    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        npimg = array(t[i].numpy(), float32)
        npimg = (npimg - mean(npimg)) / std(npimg)
        npimg = minimum(1, maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    tight_layout()
    show()

# plot_filters_multi_channel3

# Code snarfed from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
def plot_filters_multi_channel3(t,num_cols = 12,heading=''):
    ts = t.shape # 20,4,1,5,5
    num_kernels     = t.shape[0]
    num_in_channels = t.shape[2]
    num_groups      = num_kernels//num_cols
    if num_groups * num_cols < num_kernels:
        num_groups +=1
    num_rows = num_groups
    fig      = figure(figsize=(num_cols,num_rows))
    for i in range(num_kernels):
        ax1       = fig.add_subplot(num_rows,num_cols,i+1)
        npimg     = array(t[i].numpy(), float32)
        npimg     = (npimg - mean(npimg)) / std(npimg)
        npimg     = minimum(1, maximum(0, (npimg + 0.5)))
        a,b,nx,ny = npimg.shape
        npimg2    = npimg[0,0,:,:]

        ax1.imshow(npimg2)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    suptitle(heading)
    tight_layout()

# plot_weights
# Code snarfed from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e

def plot_weights(model, layer_num, single_channel = True, collated = False, heading=''):

    layer = model.features[layer_num]

    if isinstance(layer, Conv2d):
        weight_tensor = model.features[layer_num].weight.data

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor)
            else:
                plot_filters_single_channel(weight_tensor)

        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor)
            else:
                print("Can only plot weights with three channels with single channel = False")

    elif isinstance(layer, Conv3d):
        weight_tensor = model.features[layer_num].weight.data
        plot_filters_multi_channel3(weight_tensor,heading=heading)
    else:
        print("Can only visualize layers which are convolutional")

def load_net(checkpoint_file_name):
    model      = Net3C([0,0,0])
    checkpoint = load(checkpoint_file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__=='__main__':
    parser = ArgumentParser('Visualize Network')
    parser.add_argument('--chks',
                        default = 'chks',
                        help    = 'Folder for checkpoint files')
    parser.add_argument('--checkpoint',
                        default = 'monday9',
                        help    = 'Base of name for checkpoint files')
    parser.add_argument('--save',
                        default = None,
                        help    = 'Base of name for checkpoint files')
    parser.add_argument('--layer',
                        default = 0,
                        type    = int,
                        help    = 'Layer to plot')
    args = parser.parse_args()

    plot_weights(
        load_net(
            join(args.chks,f'{args.checkpoint}.pth')),
                 args.layer,
                 single_channel = False,
                 heading=f'Filters for {args.checkpoint}, layer={args.layer}')

    savefig(f'{args.checkpoint}-{args.layer}.png' if args.save==None else f'{args.save.png}')
    show()
