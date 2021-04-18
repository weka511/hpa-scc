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
from matplotlib.pyplot import figure, show, savefig, tight_layout
from random            import sample, seed
from utils             import create_xkcd_colours, set_random_seed, Timer

if __name__=='__main__':
    parser = ArgumentParser('Construct worklist, the list of images to be processed')
    parser.add_argument('--multiplets',
                        default = False,
                        action = 'store_true',
                        help   = 'Process multiplets only')
    parser.add_argument('--N',
                        type    = int,
                        default = 25,
                        help    = 'Number of each type')
    parser.add_argument('--seed',
                        default = None,
                        type    = int,
                        help    = 'Used to seed random number generator')
    parser.add_argument('--worklist',
                        default = 'worklist',
                        help    = 'Name of output file with list of images')
    parser.add_argument('--show',
                        default = False,
                        action  = 'store_true',
                        help    = 'Display histpgram')

    args                  = parser.parse_args()
    with Timer(), open(f'{args.worklist}.csv','w') as worklist:
        Descriptions          = read_descriptions()
        image_ids_by_label    = [[] for _ in Descriptions.keys()]
        for image_id,labels in read_training_expectations().items():
            if len(labels)==1 or args.multiplets:
                for label in labels:
                    image_ids_by_label[label].append(image_id)

        set_random_seed(args.seed)

        counts    = []
        for label in range(len(image_ids_by_label)):
            if args.N<len(image_ids_by_label[label]):
                for image_id in sample(image_ids_by_label[label],args.N):
                    worklist.write(f'{image_id}\n')
                counts.append(args.N)
            else:
                for image_id in image_ids_by_label[label]:
                    worklist.write(f'{image_id}\n')
                counts.append(len(image_ids_by_label[label]))

        XKCD = [colour for colour in create_xkcd_colours()][::-1]

        fig  = figure(figsize=(21,14))
        axs  = fig.subplots(ncols = 1)

        axs.bar(range(len(counts)), counts,color=XKCD[:len(counts)])
        axs.set_title(f'Number of examples for each label in {args.worklist}')
        axs.set_xticks(range(len(counts)))
        axs.set_xticklabels([Descriptions[i] for i in range(len(counts))],rotation=80)

        tight_layout()
        savefig(f'{args.worklist}.png')

    if args.show:
        show()
