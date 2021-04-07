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

from argparse import ArgumentParser
from csv      import reader
from os       import environ
from os.path  import join

# read_descriptions
#
# Read descriptions of each label
#
# Parameters:
#     file_name  Location of file contai in descriptions

def read_descriptions(file_name='descriptions.csv'):
    with open(file_name) as descriptions_file:
        return {int(row[0]) : row[1] for row in  reader(descriptions_file)}


# read_training_expectations
#
# Read list of training dataset image_ids and labels
#
# Parameters:
#     path      LOcation of training dataset
#     file_name Training dataset

def read_training_expectations(path=join(environ['DATA'],'hpa-scc'),file_name='train.csv'):
    with open(join(path,file_name)) as train:
        rows = reader(train)
        next(rows)
        return {row[0] : list(set([int(label) for label in row[1].split('|')])) for row in rows}

# read_worklist
#
# Read list of imagids that are to be processed
#
# Parameters:
#     worklist_name  Base name of file containing imageids
#     suffix         Extension

def read_worklist(worklist_name,suffix='csv'):
    with open(f'{worklist_name}.{suffix}') as worklist:
        for row in reader(worklist):
            yield row[0]
