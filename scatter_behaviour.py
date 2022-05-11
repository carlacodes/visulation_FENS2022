import seaborn as sns
import numpy as np
import mat73
import os
import scipy.io as rd
import matplotlib.pyplot as plt
import pandas as pd


bin_folder='D:/Data/Results/L28general'
fname='scatter_hit_data.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)
#matversion_difference=info_barplot['differencemat
# mean_hit_data=(info_barplot['meanABC'])
# mean_fa_data=(info_barplot['meanABCFA'])
ax=sns.scatterplot(x=[0,1,2,3], y=info_barplot['combinedmat'][0], size=100, x_jitter=2)
ax=sns.scatterplot(x=[0,1,2,3], y=info_barplot['combinedmat'][1], size=100, x_jitter=2)
ax=sns.scatterplot(x=[0,1,2,3], y=info_barplot['combinedmat'][2], size=20, x_jitter=2)
ax=sns.scatterplot(x=[0,1,2,3], y=info_barplot['combinedmat'][3], size=20, x_jitter=2)



plt.show()