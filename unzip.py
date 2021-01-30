#   Copyright (C) 2021 Greenweaves Software Limited

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import zipfile
import os

for filename in os.listdir(r'C:\data\hpa-scc'):
    if filename.endswith(".zip"):
        with zipfile.ZipFile(rf'C:\data\hpa-scc\{filename}', 'r') as zipref:
            zipref.extract(filename[0:-4],path=r'\data\hpa-scc')
        os.remove(rf'C:\data\hpa-scc\{filename}')