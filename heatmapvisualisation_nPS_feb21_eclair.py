import scipy.io as rd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

bin_folder='D:/Data/Results/decoderResults/figures/heatmaps/'
fname='confusion_matrix_eclair_nPSfeb21.mat'
info_heatmap = rd.loadmat(bin_folder + os.sep + fname)
fname_raw='confusion_matrix_raw_eclair_nPSfeb21.mat'
info_heatmap_raw=rd.loadmat(bin_folder + os.sep + fname_raw)

heatmap_bychan=info_heatmap['reorgCellbyChan'][0]
heatmap_bychan_raw=info_heatmap_raw['reorgCellbyChan'][0]
str=['craft', "in contrast to", "when a", "accurate", "rev instruments", "of science", "pink noise instruments", 'instruments']

channel_list=np.array([3, 4, 8, 9, 10, 11, 15, 16, 17, 18, 19, 22, 23, 24, 25, 27, 29, 30])
for i in range(0, len(heatmap_bychan)):
    selected_rect_data_raw=(heatmap_bychan_raw[i])

    selected_rect_data=heatmap_bychan[i]
    index_x, index_y=np.where((selected_rect_data<=0))
    show_annot_array = selected_rect_data >0
    for k in range(0, len(index_x)):
        index_x_selec=index_x[k]
        index_y_s=index_y[k]
        selected_rect_data_raw[index_x_selec][index_y_s]=np.empty(1)
    #selected_rect_data_raw=np.array2string(selected_rect_data_raw, formatter={'float_kind': lambda x: "%.2f" % x})

    ax=sns.heatmap(selected_rect_data, xticklabels=str, yticklabels=str, cmap='Oranges',annot=selected_rect_data_raw, cbar_kws={'label': 'significance score'}, vmin=0, vmax=0.3)
    for text, show_annot in zip(ax.texts, (element for row in show_annot_array for element in row)):
        text.set_visible(show_annot)

    plt.title('Eclair Original F0 Intra-trial, Site Number '+(np.array2string(channel_list[i])))
    plt.savefig(bin_folder + '\seabornheatmap_0407022_ECLAIR_feb21l27_nPS_chan'+np.array2string(channel_list[i])+'.png', dpi=500, bbox_inches='tight')

    plt.show()
