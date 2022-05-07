import scipy.io as rd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

bin_folder='D:/Data/Results/decoderResults/figures/heatmaps/'
fname='confusion_matrix.mat'
info_heatmap = rd.loadmat(bin_folder + os.sep + fname)
fname_raw='confusion_matrix_raw.mat'
info_heatmap_raw=rd.loadmat(bin_folder + os.sep + fname_raw)

heatmap_bychan=info_heatmap['reorgCellbyChan'][0]
heatmap_bychan_raw=info_heatmap_raw['reorgCellbyChan'][0]
str = ['craft', 'in contrast to', 'when a', 'accurate', 'rev. target', 'of science', 'pink noise', 'v2 in contrast to',
       'instruments']
channel_list=np.array([7,    8  ,   9  ,  10  ,  12   , 14    ,16   , 17  ,  26   , 27 ,   28  ,  30])
for i in range(0, len(heatmap_bychan)):
    selected_rect_data_raw=heatmap_bychan_raw[i]
    selected_rect_data=heatmap_bychan[i]
    ax=sns.heatmap(selected_rect_data, xticklabels=str, yticklabels=str, cmap='Blues',annot=selected_rect_data_raw, vmin=0, vmax=0.2)
    plt.title('Site Number '+(np.array2string(channel_list[i])))
    plt.show()
