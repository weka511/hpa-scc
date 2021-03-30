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
#  Compute statistics for number of images for combinations of classes

from   dirichlet         import read_training_expectations
from   matplotlib.pyplot import show, figure
from   utils             import Timer


if __name__=='__main__':
    with Timer():
        counts   = []
        for item_id,classes in read_training_expectations().items():
            class_key = '-'.join(str(c) for c in sorted(classes))
            while len(counts)<len(classes)+1:
                counts.append({})
            if class_key in counts[len(classes)]:
                counts[len(classes)][class_key] += 1
            else:
                counts[len(classes)][class_key] = 1

        fig = figure(figsize=(20,20))
        axs = fig.subplots(ncols = 2)
        axs[0].bar(range(len(counts)), [len(data) for data in counts])
        axs[0].set_title('Number of data combinations of each multiplicity')
        axs[0].set_xlabel('Multiplicty')
        axs[0].set_ylabel('Number')

        for i in range(len(counts)):
            if len(counts[i])>1:
                axs[1].hist([count for _,count in counts[i].items()],label=f'{i}',alpha=0.5)

        axs[1].legend(title='Number of classes')
        axs[1].set_title('Number of slides by combination of classes')

    show()
