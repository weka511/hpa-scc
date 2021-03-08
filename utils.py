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
from os.path import exists, join, splitext
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

class Logger:
    def __init__(self,prefix='log',suffix='.csv',logdir='./logs'):
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
        self.logfile = open(self.logfile_path,'w')
        return self

    def log(self,text):
        print (text)
        self.logfile.write(f'{text}\n')
        self.logfile.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type!=None:
            self.log(f'{exc_type}')
            self.log(f'{exc_val}')
        self.logfile.close()

if __name__=='__main__':
    with Timer('Test') as timer, Logger(prefix='foo') as log:
        x=1

