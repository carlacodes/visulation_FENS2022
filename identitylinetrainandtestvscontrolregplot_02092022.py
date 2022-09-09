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

fname='proxmat_raw0209.mat'

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
                   columns=['original F0', 'trained on original F0, tested on rove F0'])
l27zolastddev=df2.mean()
df_crumble=pd.DataFrame(arraytoplot_crumble, columns=['original F0', 'trained on original F0, tested on rove F0'])
df_eclair=pd.DataFrame(arraytoplot_eclair, columns=['original F0', 'trained on original F0, tested on rove F0'])

ax=sns.regplot(data=df2, x="original F0", y="trained on original F0, tested on rove F0", label='F1702, trained', color='cornflowerblue')
#ax=sns.pointplot(data=df2.mean(), x="original F0", y="trained on original F0, tested on rove F0", label='F1702, trained mean',ci='sd')

plt.errorbar(x=df2.mean()[0], y=df2.mean()[1], xerr=df2.std()[0], yerr=df2.std()[1], fmt='o', color='royalblue') # label='F1702 trained mean and std'

ax=sns.regplot(data=df_crumble, x="original F0", y="trained on original F0, tested on rove F0", label='F1901, naive',  color='forestgreen')
plt.errorbar(x=df_crumble.mean()[0], y=df_crumble.mean()[1], xerr=df_crumble.std()[0], yerr=df_crumble.std()[1], fmt='o', color='darkgreen')# label='F1901 naive mean and std'

ax=sns.regplot(data=df_eclair, x="original F0", y="trained on original F0, tested on rove F0", label='F1902, naive', color='mediumpurple')
plt.errorbar(x=df_eclair.mean()[0], y=df_eclair.mean()[1], xerr=df_eclair.std()[0], yerr=df_eclair.std()[1], fmt='o', color='rebeccapurple')# label='F1902 naive mean and std'


x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, 'k--', label='identity line')
ax.set(ylim=(-0.3,0.5), xlim=(-0.3, 0.5))
plt.title('Trained on Original F0, Tested on Roved F0 Scores vs. \n Original F0 Relative Decoding Scores \n (intra-trial roving)')
plt.legend()
plt.savefig(bin_folder + '\\regplot_l27_08092022origF0.png', dpi=500, bbox_inches='tight')

plt.show()
# arraytoplot=np.concatenate((matversion_zola_l27, matversion_zola_l27_control), axis=0)


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
                   columns=['original F0', 'trained on original F0, tested on rove F0'])

df_crumble=pd.DataFrame(arraytoplot_crumble, columns=['original F0', 'trained on original F0, tested on rove F0'])
df_eclair=pd.DataFrame(arraytoplot_eclair, columns=['original F0', 'trained on original F0, tested on rove F0'])

# ax=sns.regplot(data=df2, x="original F0", y="trained on original F0, tested on rove F0", label='F1702, trained', )
# #ax=sns.pointplot(data=df2.mean(), x="original F0", y="trained on original F0, tested on rove F0", label='F1702, trained mean',ci='sd')
#
# plt.errorbar(x=df2.mean()[0], y=df2.mean()[1], xerr=df2.std()[0], yerr=df2.std()[1], fmt='o', label='F1702 trained mean and std')
#
# ax=sns.regplot(data=df_crumble, x="original F0", y="trained on original F0, tested on rove F0", label='F1901, naive', )
# plt.errorbar(x=df_crumble.mean()[0], y=df_crumble.mean()[1], xerr=df_crumble.std()[0], yerr=df_crumble.std()[1], fmt='o',  label='F1901 naive mean and std')
#
# ax=sns.regplot(data=df_eclair, x="original F0", y="trained on original F0, tested on rove F0", label='F1902, naive', )
# plt.errorbar(x=df_eclair.mean()[0], y=df_eclair.mean()[1], xerr=df_eclair.std()[0], yerr=df_eclair.std()[1], fmt='o',  label='F1902 naive mean and std')
#

ax=sns.regplot(data=df2, x="original F0", y="trained on original F0, tested on rove F0", label='F1702, trained', color='cornflowerblue')
#ax=sns.pointplot(data=df2.mean(), x="original F0", y="trained on original F0, tested on rove F0", label='F1702, trained mean',ci='sd')

plt.errorbar(x=df2.mean()[0], y=df2.mean()[1], xerr=df2.std()[0], yerr=df2.std()[1], fmt='o', color='royalblue') # label='F1702 trained mean and std'

ax=sns.regplot(data=df_crumble, x="original F0", y="trained on original F0, tested on rove F0", label='F1901, naive',  color='forestgreen')
plt.errorbar(x=df_crumble.mean()[0], y=df_crumble.mean()[1], xerr=df_crumble.std()[0], yerr=df_crumble.std()[1], fmt='o', color='darkgreen')# label='F1901 naive mean and std'

ax=sns.regplot(data=df_eclair, x="original F0", y="trained on original F0, tested on rove F0", label='F1902, naive', color='mediumpurple')
plt.errorbar(x=df_eclair.mean()[0], y=df_eclair.mean()[1], xerr=df_eclair.std()[0], yerr=df_eclair.std()[1], fmt='o', color='rebeccapurple')# label='F1902 naive mean and std'



x = np.linspace(-5,5,100)
y = x
plt.plot(x, y, 'k--', label='identity line')
ax.set(ylim=(-0.3,0.5), xlim=(-0.3, 0.5))
plt.title('Trained on Original F0, Tested on Roved F0 Scores vs. \n Original F0 Relative Decoding Scores \n (Inter-trial roving)')
plt.legend()
plt.savefig(bin_folder + '\\regplot_l28_08092022origF0.png', dpi=500, bbox_inches='tight')

plt.show()

