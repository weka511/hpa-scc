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

rm(list=ls())
setwd(dirname(parent.frame(2)$ofile))
df = read.csv(file='./logs/dirichlet13.csv')
colnames(df) = c('name','N')
df <- na.omit(df)
hist(df$N,xlab='Number of Iterations',main='DPmeans')
print(sprintf('Maximum Number of Iterations in %d trials = %d', nrow(df), max(df$N)),quote = FALSE)
