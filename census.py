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
#
#  Compute statistics for number of images for combinations of labels

from hpascc            import read_descriptions,read_training_expectations
from matplotlib.pyplot import show, figure, savefig, tight_layout
from utils             import Timer,create_xkcd_colours


if __name__=='__main__':
    with Timer():
        Descriptions = read_descriptions('descriptions.csv')
        counts       = []
        label_counts = [0]*len(Descriptions)
        for item_id,labels in read_training_expectations().items():
            for label in labels:
                label_counts[label]+=1
            label_key = '-'.join(str(c) for c in sorted(labels))
            while len(counts)<len(labels)+1:
                counts.append({})
            if label_key in counts[len(labels)]:
                counts[len(labels)][label_key] += 1
            else:
                counts[len(labels)][label_key] = 1

        XKCD = [colour for colour in create_xkcd_colours()][::-1]
        fig  = figure(figsize=(21,14))
        axs  = fig.subplots(ncols = 3)

        axs[0].bar(range(len(counts)), [len(data) for data in counts],color='xkcd:teal')
        axs[0].set_title('Number of data combinations of each multiplicity')
        axs[0].set_xlabel('Multiplicity')
        axs[0].set_ylabel('Count')

        for i in range(len(counts)):
            if len(counts[i])>1:
                axs[1].hist([count for _,count in counts[i].items()],label=f'{i}',
                            alpha = 0.5,
                            color = XKCD[i-1])

        axs[1].legend(title='Number of labels')
        axs[1].set_title('Number of cells by combination of labels')

        x = list(range(len(label_counts)))
        axs[2].bar(x,label_counts, color=[XKCD[i] for i in x])
        axs[2].set_title('Number of examples for each label')
        axs[2].set_xticks(x)
        axs[2].set_xticklabels([Descriptions[i] for i in x],rotation=90)

        tight_layout()
        savefig('census.png')

    show()
