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

from argparse          import ArgumentParser
from hpascc            import read_training_expectations, read_descriptions
from matplotlib.pyplot import figure, show, savefig
from random            import sample, seed
from utils             import set_random_seed, create_xkcd_colours

if __name__=='__main__':
    parser = ArgumentParser('Select Data')
    parser.add_argument('--multiplets',
                        default = False,
                        action = 'store_true',
                        help   = 'Process multiplets only')
    parser.add_argument('--N',
                        type    = int,
                        default = 2,
                        help    = 'Number of each type')
    parser.add_argument('--seed',
                        default = None,
                        type    = int,
                        help    = 'Used to seed random number generator')
    args                  = parser.parse_args()
    Descriptions          = read_descriptions()
    image_ids_by_label    = [[] for _ in Descriptions.keys()]
    for image_id,labels in read_training_expectations().items():
        if len(labels)==1 or args.multiplets:
            for label in labels:
                image_ids_by_label[label].append(image_id)

    set_random_seed(args.seed)
    selection = []
    counts    = []
    for label in range(len(image_ids_by_label)):
        if args.N<len(image_ids_by_label[label]):
            for image_id in sample(image_ids_by_label[label],args.N):
                selection.append(image_id)
            counts.append(args.N)
        else:
            for image_id in image_ids_by_label[label]:
                selection.append(image_id)
            counts.append(len(image_ids_by_label[label]))
    XKCD = [colour for colour in create_xkcd_colours()][::-1]

    fig  = figure(figsize=(21,14))
    axs  = fig.subplots(ncols = 2)

    axs[0].bar(range(len(counts)), counts,color=XKCD[:len(counts)])

    show()
