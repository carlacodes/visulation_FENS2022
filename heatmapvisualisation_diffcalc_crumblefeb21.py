import scipy.io as rd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

animalid=['Crumble', 'Eclair']
for i0 in animalid:
    bin_folder = 'D:/Data/Results/decoderResults/figures/heatmaps/'

    if i0=='Crumble':
        fname='confusion_matrix_crumblefeb21_PS.mat'
        info_heatmap = rd.loadmat(bin_folder + os.sep + fname)
        fname_raw='confusion_matrix_raw_crumblefeb21_PS.mat'
        info_heatmap_raw=rd.loadmat(bin_folder + os.sep + fname_raw)

        fname_nps = 'confusion_matrix_crumblefeb21_nPS.mat'
        info_heatmap_nps = rd.loadmat(bin_folder + os.sep + fname_nps)
        fname_raw_nps = 'confusion_matrix_raw_crumblefeb21_nPS.mat'
        info_heatmap_raw_nps = rd.loadmat(bin_folder + os.sep + fname_raw_nps)
        channel_list = np.array([6, 7, 10, 12, 13, 16, 18, 19, 20, 22, 23, 24, 26])

    if i0=='Eclair':
        fname = 'confusion_matrix_eclair_PSfeb21.mat'
        info_heatmap = rd.loadmat(bin_folder + os.sep + fname)
        fname_raw = 'confusion_matrix_raw_eclair_PSfeb21.mat'
        info_heatmap_raw = rd.loadmat(bin_folder + os.sep + fname_raw)

        fname_nps = 'confusion_matrix_eclair_nPSfeb21.mat'
        info_heatmap_nps = rd.loadmat(bin_folder + os.sep + fname_nps)
        fname_raw_nps = 'confusion_matrix_raw_eclair_nPSfeb21.mat'
        info_heatmap_raw_nps = rd.loadmat(bin_folder + os.sep + fname_raw_nps)
        channel_list = np.array([3, 4, 8, 9, 10, 11, 15, 16, 17, 18, 19, 22, 23, 24, 25, 27, 29, 30])

    heatmap_bychan=info_heatmap['reorgCellbyChan'][0]
    heatmap_bychan_raw=info_heatmap_raw['reorgCellbyChan'][0]
    str=['craft', "in contrast to", "when a", "accurate", "rev instruments", "of science", "pink noise instruments", 'instruments']





    heatmap_bychan_nps=info_heatmap_nps['reorgCellbyChan'][0]
    heatmap_bychan_raw_nps=info_heatmap_raw_nps['reorgCellbyChan'][0]
    str=['craft', "in contrast to", "when a", "accurate", "rev instruments", "of science", "pink noise instruments", 'instruments']


    difference_between_roving=np.subtract(heatmap_bychan_raw_nps,heatmap_bychan_raw)


    for i in range(0, len(heatmap_bychan)):
        selected_rect_data_raw=(heatmap_bychan_raw[i])
        selected_rect_data_raw_nps=(heatmap_bychan_raw_nps[i])

        selected_rect_data=heatmap_bychan[i]
        selected_rect_data_nps=heatmap_bychan_nps[i]

        relative_difference=np.subtract(selected_rect_data,selected_rect_data_nps)
        sns.histplot(relative_difference)
        plt.xlim(-0.2, 0.2)
        plt.title(i0+' '+(np.array2string(channel_list[i])))
        plt.xlabel('Rove F0 Score - Original F0 Score')

        # index_x, index_y=np.where((selected_rect_data<=0))
        # show_annot_array = selected_rect_data >0
        # for k in range(0, len(index_x)):
        #     index_x_selec=index_x[k]
        #     index_y_s=index_y[k]
        #     selected_rect_data_raw[index_x_selec][index_y_s]=np.empty(1)
        # #selected_rect_data_raw=np.array2string(selected_rect_data_raw, formatter={'float_kind': lambda x: "%.2f" % x})
        #
        # ax=sns.heatmap(selected_rect_data, xticklabels=str, yticklabels=str, cmap='Oranges',annot=selected_rect_data_raw, cbar_kws={'label': 'significance score'}, vmin=0, vmax=0.2)
        # for text, show_annot in zip(ax.texts, (element for row in show_annot_array for element in row)):
        #     text.set_visible(show_annot)
        #
        # plt.title('Crumble Roved F0, Intra-trial roving, Site Number '+(np.array2string(channel_list[i])))
        plt.savefig(bin_folder + '\seabornhistogram_07072022_L27'+i0+np.array2string(channel_list[i])+'.png', dpi=500, bbox_inches='tight')

        plt.show()
