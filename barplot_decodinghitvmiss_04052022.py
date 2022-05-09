import seaborn as sns
import numpy as np
import mat73
import os
import scipy.io as rd
import matplotlib.pyplot as plt
import pandas as pd


bin_folder='D:/Data/Results/decoderResults/figures/OriginalDataFigures'
fname='barchartresults.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)
matversion_difference=info_barplot['differencemat']
# matversion_crumble=info_barplot['matversion_crumble']
# matversion_crumble_L27=info_barplot['matversion_crumble_l27']
# matversion_eclair=info_barplot['matversion_eclair']
# matversion_eclair_l27=info_barplot['matversion_eclair_l27']
# matversion_zola=info_barplot['matversion_zola']
# matversion_zola_l27=info_barplot['matversion_zola_l27']
# tips = sns.load_dataset("tips")
#
# ax = plt.subplots()
# #zola=bar(mean(matversion_zola(:, [2,1,3])))
#
# zola_y=np.mean(matversion_zola, axis=0)
# zola_y_l27=np.mean(matversion_zola_l27, axis=0)
# zola_y_1=(zola_y[1])
# zola_y_2=(zola_y[0])
# zola_y_3=(zola_y[2])
# zola_y_l28=[zola_y_1, zola_y_2, zola_y_3]
# zola_y_l28_scatter=[matversion_zola[:,1], matversion_zola[:,0], matversion_zola[:,2]]
# zola_y_l27_bar=[zola_y_l27[0], zola_y_l27[7], zola_y_l27[2],  zola_y_l27[3],  zola_y_l27[4],  zola_y_l27[5],  zola_y_l27[6]]
# zola_y_l27_scatter=[matversion_zola_l27[:, 0], matversion_zola_l27[:, 7], matversion_zola_l27[:,2],  matversion_zola_l27[:,3],  matversion_zola_l27[:,4],  matversion_zola_l27[:,5],  matversion_zola_l27[:,6]]
#[1,8,3:7]
# plotting columns
# ax=sns.stripplot(data=zola_y_l28_scatter, jitter=0.1, color='green', alpha=0.8)
# ax=sns.stripplot(data=matversion_crumble, jitter=0.1, color='purple', alpha=0.8)
# ax=sns.stripplot(data=matversion_eclair, jitter=0.1, color='navy', alpha=0.8)
# ax = sns.barplot([1,2,3], zola_y_l28,  color='green', label='F1702',alpha=.3)
# ax = sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_eclair, axis=0), color='navy', alpha=0.3, label='F1902')
# ax=sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_crumble, axis=0),  color='purple',  alpha=0.3, label='F1901')
# x_vec=np.empty((24,2))
# ax=sns.boxplot( y=matversion_difference[0])
# x_vecc1=np.transpose(np.array([0] * 12))
# x_vecc2=np.transpose(np.array([1] * 12))
# np.reshape(x_vecc1, (1,12), order='C')
# x_vec[0]=np.transpose(np.concatenate((x_vecc1, x_vecc2), axis=0))
# x_vec[1]=np.concatenate((matversion_difference[:,0], matversion_difference[:, 1]), axis=0)
#
# x_vec=x_vec.tolist()
df = pd.DataFrame(matversion_difference, columns = ['Rove','Natural'])

ax=sns.boxplot(y=df["Rove"])
plt.title('Decoding score of behavioural response using data post-linear shift model', fontsize=10)
str={'roved F0', 'control F0'}
plt.xticks([0,1], labels=str, rotation=45)
plt.xlabel('F0 Type of Target Stimulus')
plt.ylabel('Relative decoding score (score-chance)')

plt.legend(fontsize=8)
plt.ylim([-0.35, 0.4])
ax = sns.boxplot(data=df, orient="y", palette="magma")
ax=sns.stripplot(data=df, jitter=0.1, palette='husl', alpha=1)
plt.savefig(bin_folder + '\seabornboxplotmissvhit_l27.png', bbox_inches='tight')

plt.show()


#barchartresults_orig
fname='barchartresults_orig.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)
matversion_difference_orig=info_barplot['differencemat']

df_orig = pd.DataFrame(matversion_difference_orig, columns = ['Rove','Natural'])

plt.title('Decoding score of behavioural response using original data', fontsize=10)
str={'roved F0', 'control F0'}
plt.xticks([0,1], labels=str, rotation=45)
plt.xlabel('F0 Type of Target Stimulus')
plt.ylabel('Relative decoding score (score-chance)')

plt.legend(fontsize=8)
plt.ylim([-0.35, 0.4])
ax = sns.boxplot(data=df_orig, orient="y", palette="magma")
ax=sns.stripplot(data=df_orig, jitter=0.1, palette='husl', alpha=1)
plt.savefig(bin_folder + '\seabornboxplothitvsmiss_origdata_l27.png', bbox_inches='tight')

plt.show()


#ax2 = plt.subplots()
# ax2=sns.stripplot(data=zola_y_l27_scatter, jitter=0.1, color='green', alpha=0.8)
# ax2=sns.stripplot(data=matversion_crumble_L27, jitter=0.1, color='purple', alpha=0.8)
# ax2=sns.stripplot(data=matversion_eclair_l27, jitter=0.1, color='navy', alpha=0.8)
# ax2 = sns.barplot([1,2,3,4,5,6,7], zola_y_l27_bar,  color='green', label='F1702',alpha=.3)
# ax2 = sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_eclair_l27, axis=0), color='navy', alpha=0.3, label='F1902')
# ax2=sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_crumble_L27, axis=0),  color='purple',  alpha=0.3, label='F1901')
# plt.xlabel('Distractor')
# str={'craft', "in contrast to", "when a", "accurate", "rev instruments", "of science", "pink noise instruments"}
# plt.xticks([0,1,2,3,4,5,6], labels=str, rotation=45)
# plt.ylabel('Change in Mean Decoding Score for All Sites')
#
# plt.title('Change in decoding score between train and test dataset, intra-trial roving',fontsize=10)
#
# plt.legend(fontsize=8)
# plt.ylim([-0.35, 0.35])
# plt.savefig(bin_folder + '\seabornbarplot_l27.png', dpi=500, bbox_inches='tight')
# plt.show()
