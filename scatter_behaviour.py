import seaborn as sns
import numpy as np
import mat73
import os
import scipy.io as rd
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

bin_folder='D:/Data/Results/L28general'
fname='scatter_hit_data.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)
#matversion_difference=info_barplot['differencemat
# mean_hit_data=(info_barplot['meanABC'])
# mean_fa_data=(info_barplot['meanABCFA'])
xA, xB = np.random.normal(0, 0.1, len(info_barplot['combinedmat'][0])), np.random.normal(1, 0.1, len(info_barplot['combinedmat'][0]))
xC, xD = np.random.normal(2, 0.1, len(info_barplot['combinedmat'][0])), np.random.normal(3, 0.1, len(info_barplot['combinedmat'][0]))
#
#
# ax=sns.stripplot(x=[0,1,2,3], y=info_barplot['combinedmat'][0], s=10, jitter=0.2)
# ax=sns.stripplot(x=[0,1,2,3], y=info_barplot['combinedmat'][1], s=10, jitter=0.2)
# ax=sns.stripplot(x=[0,1,2,3], y=info_barplot['combinedmat'][2], s=10, jitter=0.2)
# ax=sns.stripplot(x=[0,1,2,3], y=info_barplot['combinedmat'][3], s=10, jitter=0.2)
xA=[0.03,1.03,2.03,3.03]
ax=plt.errorbar([0,1,2,3], info_barplot['combinedmat'][0], yerr=info_barplot['combinedmatSD'][0],fmt="o", elinewidth=2, ms=5, color='royalblue', label='Intra-trial Roved F0')
#ax=plt.scatter([0,1,2,3], info_barplot['combinedmat'][0], s=100, facecolor='blue')
for i, txt in enumerate( info_barplot['combinedMatCatch'][0]):
    plt.annotate(txt, (xA[i],info_barplot['combinedmat'][0][i]))


ax=plt.errorbar([0.1,1.1,2.1,3.1], info_barplot['combinedmat'][1], yerr=info_barplot['combinedmatSD'][1],fmt="o", elinewidth=2, ms=5, color='cornflowerblue', label='Control F0')
xB=[0.13,1.13,2.13,3.13]
for i, txt in enumerate( info_barplot['combinedMatCatch'][1]):
    plt.annotate(txt, (xB[i],info_barplot['combinedmat'][1][i]))

xC=[0.33,1.33,2.33,3.33]

ax=plt.errorbar([0.3,1.3,2.3,3.3], info_barplot['combinedmat'][2], yerr=info_barplot['combinedmatSD'][2],fmt="o", elinewidth=2, ms=5, color='lightgreen', label='Inter-trial Roved F0')
for i, txt in enumerate( info_barplot['combinedMatCatch'][2]):
    plt.annotate(txt, (xC[i],info_barplot['combinedmat'][2][i]))

xD=[0.43,1.43,2.43,3.43]
ax=plt.errorbar([0.4,1.4,2.4,3.4], info_barplot['combinedmat'][3], yerr=info_barplot['combinedmatSD'][3],fmt="o", elinewidth=2, ms=5, color='forestgreen', label='Control F0')
for i, txt in enumerate( info_barplot['combinedMatCatch'][3]):
    plt.annotate(txt, (xD[i],info_barplot['combinedmat'][3][i]))


x = np.linspace(0,3.5,100)
y = [0.33] * 100
plt.plot(x, y,'--y',label='Chance')
plt.ylim([0, 1])
plt.xticks([0,1,2,3], labels=['F1702 Zola', 'F1815 Cruella', 'F1803 Tina*', 'F2002 Macaroni*'])
plt.legend(fontsize=8)
plt.xlabel('Ferret Identity', fontsize=15)
plt.ylabel('P. of Hits', fontsize=15)
plt.title('Proportion of Correct Responses for \n Intra and Inter-trial F0 Roving', fontsize=20)
plt.savefig(bin_folder + '\seaborn_scatter_hits120522.png', dpi=500, bbox_inches='tight')

plt.show()