import seaborn as sns
import numpy as np
import mat73
import os
import scipy.io as rd
import pandas as pd
import matplotlib.pyplot as plt


bin_folder='D:\Data\Results\InvarianceScores'
fname='proxmat_raw2208.mat'

info_barplot = rd.loadmat(bin_folder + os.sep + fname)

matversion_crumble=info_barplot['matversion_crumble']
matversion_crumble_L27=info_barplot['matversion_crumble_l27']
matversion_eclair=info_barplot['matversion_eclair']
matversion_eclair_l27=info_barplot['matversion_eclair_l27']
matversion_zola=info_barplot['matversion_zola']
matversion_zola_l27=info_barplot['matversion_zola_l27']

fname='proxmat_raw0209PS.mat'

info_barplot_control = rd.loadmat(bin_folder + os.sep + fname)

matversion_crumble_control=info_barplot_control['matversion_crumble']
matversion_crumble_L27_control=info_barplot_control['matversion_crumble_l27']
matversion_eclair_control=info_barplot_control['matversion_eclair']
matversion_eclair_l27_control=info_barplot_control['matversion_eclair_l27']
matversion_zola_control=info_barplot_control['matversion_zola']
matversion_zola_l27_control=info_barplot_control['matversion_zola_l27']

#flatten for plotting
matversion_zola_l27_2=matversion_zola_l27.flatten()
matversion_zola_l27_control_2=matversion_zola_l27_control.flatten()

matversion_crumble_L27_control=matversion_crumble_L27_control.flatten()
matversion_crumble_L27=matversion_crumble_L27.flatten()

matversion_eclair_l27_control=matversion_eclair_l27_control.flatten()
matversion_eclair_l27=matversion_eclair_l27.flatten()


arraytoplot = np.zeros(((matversion_zola_l27_control.size),2))

arraytoplot_crumble = np.zeros(((matversion_crumble_L27_control.size),2))
arraytoplot_eclair=np.zeros(((matversion_eclair_l27.size),2))

arraytoplot[:,0]=matversion_zola_l27_control_2
arraytoplot[:,1]=matversion_zola_l27_2

arraytoplot_crumble[:,0]=matversion_crumble_L27_control
arraytoplot_crumble[:,1]=matversion_crumble_L27

arraytoplot_eclair[:,0]=matversion_eclair_l27_control
arraytoplot_eclair[:,1]=matversion_eclair_l27



df2 = pd.DataFrame(arraytoplot,
                   columns=['roved F0 score', 'trained on original F0, tested on rove F0 score'])
l27zolastddev=df2.mean()
df_crumble=pd.DataFrame(arraytoplot_crumble, columns=['roved F0 score', 'trained on original F0, tested on rove F0 score'])
df_eclair=pd.DataFrame(arraytoplot_eclair, columns=['roved F0 score', 'trained on original F0, tested on rove F0 score'])

ax=sns.scatterplot(data=df2, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained', alpha=0.7)
#ax=sns.pointplot(data=df2.mean(), x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained mean',ci='sd')

plt.errorbar(x=df2.mean()[0], y=df2.mean()[1], xerr=df2.std()[0], yerr=df2.std()[1], fmt='o', label='mean', mfc='blue',
         mec='blue')

ax=sns.scatterplot(data=df_crumble, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1901, naive', alpha=0.7)
plt.errorbar(x=df_crumble.mean()[0], y=df_crumble.mean()[1], xerr=df_crumble.std()[0], yerr=df_crumble.std()[1], fmt='o',  label='mean')

ax=sns.scatterplot(data=df_eclair, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1902, naive', alpha=0.7)
plt.errorbar(x=df_eclair.mean()[0], y=df_eclair.mean()[1], xerr=df_eclair.std()[0], yerr=df_eclair.std()[1], fmt='o',  label='mean')


x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, 'k--', label='identity')
ax.set(ylim=(-0.3,0.5), xlim=(-0.3, 0.5))
plt.title('Trained on original F0, tested on roved F0 scores vs. \n roved F0 Relative Decoding Scores (intra-trial roving)')
plt.legend()
plt.show()
# arraytoplot=np.concatenate((matversion_zola_l27, matversion_zola_l27_control), axis=0)

ax=sns.regplot(data=df2, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained')

#ax=sns.pointplot(data=df2.mean(), x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained mean',ci='sd')

#plt.errorbar(x=df2.mean()[0], y=df2.mean()[1], xerr=df2.std()[0], yerr=df2.std()[1], fmt='o', label='F1702 trained mean and std')

ax=sns.regplot(data=df_crumble, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1901, naive')
#plt.errorbar(x=df_crumble.mean()[0], y=df_crumble.mean()[1], xerr=df_crumble.std()[0], yerr=df_crumble.std()[1], fmt='o',  label='F1901 naive mean and std')

ax=sns.regplot(data=df_eclair, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1902, naive')
#plt.errorbar(x=df_eclair.mean()[0], y=df_eclair.mean()[1], xerr=df_eclair.std()[0], yerr=df_eclair.std()[1], fmt='o',  label='F1902 naive mean and std')
x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, 'k--', label='identity line')
ax.set(ylim=(-0.3,0.5), xlim=(-0.3, 0.5))
plt.title('Trained on original F0, Tested on Roved F0 Scores vs. \n Roved F0 Relative Decoding Scores \n (intra-trial roving)')
plt.legend()
plt.savefig(bin_folder + '\\regplot_l27_03092022.png', dpi=500, bbox_inches='tight')

plt.show()


##L28

matversion_zola_L28_2=matversion_zola.flatten()
matversion_zola_L28_control_2=matversion_zola_control.flatten()

matversion_crumble_L28_control=matversion_crumble_control.flatten()
matversion_crumble_L28=matversion_crumble.flatten()

matversion_eclair_L28_control=matversion_eclair_control.flatten()
matversion_eclair_L28=matversion_eclair.flatten()


arraytoplot = np.zeros(((matversion_zola_L28_control_2.size),2))

arraytoplot_crumble = np.zeros(((matversion_crumble_L28_control.size),2))
arraytoplot_eclair=np.zeros(((matversion_eclair_L28.size),2))

arraytoplot[:,0]=matversion_zola_L28_control_2
arraytoplot[:,1]=matversion_zola_L28_2

arraytoplot_crumble[:,0]=matversion_crumble_L28_control
arraytoplot_crumble[:,1]=matversion_crumble_L28

arraytoplot_eclair[:,0]=matversion_eclair_L28_control
arraytoplot_eclair[:,1]=matversion_eclair_L28



df2 = pd.DataFrame(arraytoplot,
                   columns=['roved F0 score', 'trained on original F0, tested on rove F0 score'])

df_crumble=pd.DataFrame(arraytoplot_crumble, columns=['roved F0 score', 'trained on original F0, tested on rove F0 score'])
df_eclair=pd.DataFrame(arraytoplot_eclair, columns=['roved F0 score', 'trained on original F0, tested on rove F0 score'])

ax=sns.scatterplot(data=df2, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained', alpha=0.7)

#ax=sns.pointplot(data=df2.mean(), x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained mean',ci='sd')

plt.errorbar(x=df2.mean()[0], y=df2.mean()[1], xerr=df2.std()[0], yerr=df2.std()[1], fmt='o', label='F1702 trained mean and std')

ax=sns.scatterplot(data=df_crumble, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1901, naive', alpha=0.7)
plt.errorbar(x=df_crumble.mean()[0], y=df_crumble.mean()[1], xerr=df_crumble.std()[0], yerr=df_crumble.std()[1], fmt='o',  label='F1901 naive mean and std')

ax=sns.scatterplot(data=df_eclair, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1902, naive', alpha=0.7)
plt.errorbar(x=df_eclair.mean()[0], y=df_eclair.mean()[1], xerr=df_eclair.std()[0], yerr=df_eclair.std()[1], fmt='o',  label='F1902 naive mean and std')


x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, 'k--', label='identity line')
ax.set(ylim=(-0.3,0.5), xlim=(-0.3, 0.5))
plt.title('Trained on Original F0, Tested on Roved F0 Scores vs. \n Roved F0 Relative Decoding Scores \n (Inter-trial roving)')
plt.legend()

plt.show()


ax=sns.regplot(data=df2, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained')

#ax=sns.pointplot(data=df2.mean(), x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1702, trained mean',ci='sd')

#plt.errorbar(x=df2.mean()[0], y=df2.mean()[1], xerr=df2.std()[0], yerr=df2.std()[1], fmt='o', label='mean',mfc='blue',
         #mec='blue')

ax=sns.regplot(data=df_crumble, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1901, naive')
#plt.errorbar(x=df_crumble.mean()[0], y=df_crumble.mean()[1], xerr=df_crumble.std()[0], yerr=df_crumble.std()[1], fmt='o',  label='F1901 naive mean and std')

ax=sns.regplot(data=df_eclair, x="roved F0 score", y="trained on original F0, tested on rove F0 score", label='F1902, naive')
#plt.errorbar(x=df_eclair.mean()[0], y=df_eclair.mean()[1], xerr=df_eclair.std()[0], yerr=df_eclair.std()[1], fmt='o',  label='F1902 naive mean and std')


x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, 'k--', label='identity')
ax.set(ylim=(-0.3,0.5), xlim=(-0.3, 0.5))
plt.title('Trained on Original F0, Tested on Roved F0 Scores vs. \n Roved F0 Relative Decoding Scores (inter-trial roving)')
plt.legend()
plt.savefig(bin_folder + '\\regplot_l28_03092022.png', dpi=500, bbox_inches='tight')

plt.show()

