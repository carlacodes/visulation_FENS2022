import scipy.io as rd
import matplotlib.pyplot as plt
import os
import seaborn as sns

bin_folder='D:/Data/Results/decoderResults/figures/heatmaps/'
fname='confusion_matrix.mat'
info_heatmap = rd.loadmat(bin_folder + os.sep + fname)
fname_raw='confusion_matrix_raw.mat'
info_heatmap_raw=rd.loadmat(bin_folder + os.sep + fname)

heatmap_bychan=info_heatmap['reorgCellbyChan'][0]

for i in range(0, len(heatmap_bychan)):
    selected_rect_data=heatmap_bychan[i]
    ax=sns.heatmap(selected_rect_data, cmap='Blues', vmin=0, vmax=0.2)
    plt.show()
