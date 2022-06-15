import seaborn as sns
import numpy as np
import mat73
import os
import scipy.io as rd
import matplotlib.pyplot as plt


bin_folder='D:\Data\Results\InvarianceScores'
fname='proxmat_raw.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)

matversion_crumble=info_barplot['matversion_crumble']
matversion_crumble_L27=info_barplot['matversion_crumble_l27']
matversion_eclair=info_barplot['matversion_eclair']
matversion_eclair_l27=info_barplot['matversion_eclair_l27']
matversion_zola=info_barplot['matversion_zola']
matversion_zola_l27=info_barplot['matversion_zola_l27']
tips = sns.load_dataset("tips")

ax = plt.subplots()
#zola=bar(mean(matversion_zola(:, [2,1,3])))

zola_y=np.mean(matversion_zola, axis=0)
zola_y_l27=np.mean(matversion_zola_l27, axis=0)
zola_y_1=(zola_y[1])
zola_y_2=(zola_y[0])
zola_y_3=(zola_y[2])
zola_y_l28=[zola_y_1, zola_y_2, zola_y_3]
zola_y_l28_scatter=[matversion_zola[:,1], matversion_zola[:,0], matversion_zola[:,2]]
zola_y_l27_bar=[zola_y_l27[0], zola_y_l27[7], zola_y_l27[2],  zola_y_l27[3],  zola_y_l27[4],  zola_y_l27[5],  zola_y_l27[6]]
zola_y_l27_scatter=[matversion_zola_l27[:, 0], matversion_zola_l27[:, 7], matversion_zola_l27[:,2],  matversion_zola_l27[:,3],  matversion_zola_l27[:,4],  matversion_zola_l27[:,5],  matversion_zola_l27[:,6]]
#[1,8,3:7]
# plotting columns
ax=sns.stripplot(data=zola_y_l28_scatter, jitter=0.1, color='green', alpha=0.8)
ax=sns.stripplot(data=matversion_crumble, jitter=0.1, color='purple', alpha=0.8)
ax=sns.stripplot(data=matversion_eclair, jitter=0.1, color='navy', alpha=0.8)
ax = sns.barplot([1,2,3], zola_y_l28,  color='green', label='F1702, trained',alpha=.3)
ax = sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_eclair, axis=0), color='navy', alpha=0.3, label='F1902, naive')
ax=sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_crumble, axis=0),  color='purple',  alpha=0.3, label='F1901, naive')
plt.title('Change in decoding score between train and test dataset, inter-trial roving', fontsize=10)
str=['craft', "in contrast to", "when a", "accurate", "rev instruments", "of science", "pink noise instruments"]
plt.xticks([0,1,2,3,4,5,6], labels=str, rotation=45)
plt.xlabel('Distractor')
plt.ylabel('Change in Mean Decoding Score for All Sites')

plt.legend(fontsize=8)
plt.ylim([-0.35, 0.35])
plt.savefig(bin_folder + '\seabornbarplot_l28_raw.png', bbox_inches='tight')
plt.show()


ax2 = plt.subplots()
ax2=sns.stripplot(data=zola_y_l27_scatter, jitter=0.1, color='green', alpha=0.8)
ax2=sns.stripplot(data=matversion_crumble_L27, jitter=0.1, color='purple', alpha=0.8)
ax2=sns.stripplot(data=matversion_eclair_l27, jitter=0.1, color='navy', alpha=0.8)
ax2 = sns.barplot([1,2,3,4,5,6,7], zola_y_l27_bar,  color='green', label='F1702, trained',alpha=.3)
ax2 = sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_eclair_l27, axis=0), color='navy', alpha=0.3, label='F1902, naive')
ax2=sns.barplot([1,2,3,4,5,6,7], np.mean(matversion_crumble_L27, axis=0),  color='purple',  alpha=0.3, label='F1901, naive')
plt.xlabel('Distractor')

str=['craft', "in contrast to", "when a", "accurate", "rev instruments", "of science", "pink noise instruments"]
plt.xticks([0,1, 2,3,4,5,6], labels=str, rotation=45)
plt.ylabel('Change in Mean Decoding Score for All Sites')

plt.title('Change in decoding score between train and test dataset, intra-trial roving',fontsize=10)

plt.legend(fontsize=8)
plt.ylim([-0.35, 0.35])
plt.savefig(bin_folder + '\seabornbarplot_raw_l27.png', dpi=500, bbox_inches='tight')
plt.show()
