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

from os      import walk
from os.path import basename,exists, join, splitext
from random  import randrange, seed
from sys     import maxsize
from time    import time

# Timer
#
# This class is used as a wrapper when I want to know the execution time of some code.

class Timer:
    def __init__(self, message = None):
        self.start   = None
        self.message = '' if message is None else f'({message}) '

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time() - self.start
        minutes = int(elapsed/60)
        seconds = elapsed - 60*minutes
        print (f'Elapsed Time {self.message}{minutes} m {seconds:.2f} s')

# Logger
#
# Record log data in logfile

class Logger:
    # __init__
    #
    # Initializa logger
    #
    # Parameters:
    #     prefix   Logfile names are of form prefixnnn.suffix
    #     suffix   Logfile names are of form prefixnnn.suffix
    #     logdir   Directory for logfiles
    #     dummy    Used to suppress logging

    def __init__(self,prefix='log',suffix='.csv',logdir='./logs',dummy=False):
        self.dummy = dummy
        if self.dummy: return
        logs = [join(logdir,filename) for _,_,filenames in walk(logdir) \
                for filename in filenames                               \
                if filename.startswith(prefix) and splitext(filename)[-1]==suffix]
        i    = len(logs)
        self.logfile_path = join(logdir, f'{prefix}{i+1}{suffix}')
        while exists(self.logfile_path):
            i += 1
            self.logfile_path = join(logdir, f'{prefix}{i+1}{suffix}')
        self.logfile = None

    def __enter__(self):
        if self.dummy: return self
        self.logfile = open(self.logfile_path,'w')
        return self

    def log(self,text):
        if self.dummy: return
        print (text)
        self.logfile.write(f'{text}\n')
        self.logfile.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dummy: return
        if exc_type!=None:
            self.log(f'{exc_type}')
            self.log(f'{exc_val}')
        self.logfile.close()

# set_seed
#
# Seed random number so run can be reproduced. If seed is not specified
# generate a new seed and record it.
#
# Parameters:
#     specified_seed  Seed to be used (or None if random seed)
#     file_name       Name of file where actual seed will be stored
#
# Returns:
#    Seed value that was actually used

def set_random_seed(specified_seed=None,prefix='seed',suffix='txt'):
    file_name = f'{splitext(basename(prefix))[0]}.{suffix}'
    if specified_seed==None:
        seed()
        new_seed = randrange(maxsize)
        print (f'Seed = {new_seed}')
        with open(file_name,'w') as out:
            out.write(f'Seed = {new_seed}\n')
        seed(new_seed)
        return new_seed
    else:
        print (f'Reusing seed = {specified_seed}')
        seed(specified_seed)
        return specified_seed

if __name__=='__main__':
    with Timer('Test') as timer, Logger(prefix='utilstest') as log:
        set_random_seed(prefix=__file__)
