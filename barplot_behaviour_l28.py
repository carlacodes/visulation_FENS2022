import seaborn as sns
import numpy as np
import mat73
import os
import scipy.io as rd
import matplotlib.pyplot as plt
import pandas as pd


# cd('D:/Data/Results/L27general')
# set(gcf,'Position',[500 200 1500 1000])
#
# title('Proportion of Correct Centre Spout Releases Over Pitch of Target Word, Intra-Trial F0 Roving', 'FontSize',20)
# save('pitch_curve_data_L27.mat','outDatahitsF','outDatahitsM', 'SD_ABCFA','SD_ABC', 'meanABC', 'meanABCFA')
tips = sns.load_dataset("tips")

bin_folder='D:/Data/Results/L28general'
fname='pitch_curve_data_L28.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)
#matversion_difference=info_barplot['differencemat
mean_hit_data=(info_barplot['meanABC'])
mean_fa_data=(info_barplot['meanABCFA'])
individual_hit_dataf=np.transpose(info_barplot['outDatahitsF'])
individual_hit_datam=np.transpose(info_barplot['outDatahitsM'])
var_range=info_barplot['varRange']
var_range=np.transpose(var_range)
combined_array=np.concatenate([mean_hit_data, mean_fa_data, var_range], axis=1)
var_range_F=np.transpose(info_barplot['varRangeF'])
var_range_M=np.transpose(info_barplot['varRangeM'])

combined_indiv_hit_F=np.concatenate([individual_hit_dataf, var_range_F], axis=1)
combined_indiv_hit_M=np.concatenate([individual_hit_datam, var_range_M], axis=1)


fig, ax1 = plt.subplots()



info_barplot_df=pd.DataFrame(combined_array, columns=['Hit_Rate', 'False_Alarm_Rate', 'Pitch'])
colors = ['seagreen' if c >= 4 else 'mediumseagreen' for c in info_barplot_df["Pitch"]]

info_barplot_df_indivFhit=pd.DataFrame(combined_indiv_hit_F, columns=['Macaroni', 'Tina', 'Cruella', 'Zola', 'Pitch'])
info_barplot_df_indivMhit=pd.DataFrame(combined_indiv_hit_M, columns=['Macaroni', 'Tina', 'Cruella', 'Zola', 'Pitch'])


# mean_hit_data=pd.DataFrame(mean_hit_data, columns = ['1', '2', '3', '4', '5', '6'])
ax=sns.barplot(x="Pitch", y="Hit_Rate", yerr=info_barplot['SD_ABC'], data=info_barplot_df, palette=colors, alpha=0.8, ax=ax1)
ax=sns.lineplot(data=info_barplot_df_indivFhit, x='Pitch', y='Macaroni', color='royalblue', ax=ax1)
ax=sns.lineplot(data=info_barplot_df_indivFhit, x='Pitch', y='Cruella', color='darkorange')
ax=sns.lineplot(data=info_barplot_df_indivFhit, x='Pitch', y='Zola', color='orangered')
ax=sns.lineplot(data=info_barplot_df_indivFhit, x='Pitch', y='Tina',  color='pink')


ax=sns.lineplot(data=info_barplot_df_indivMhit, x='Pitch', y='Macaroni',color='cornflowerblue', ax=ax1)
ax=sns.lineplot(data=info_barplot_df_indivMhit, x='Pitch', y='Cruella', color='orange')
ax=sns.lineplot(data=info_barplot_df_indivMhit, x='Pitch', y='Zola',  color='red')
ax=sns.lineplot(data=info_barplot_df_indivMhit, x='Pitch', y='Tina',  color='hotpink')
ax2=sns.barplot(x="Pitch", y="False_Alarm_Rate", yerr=(info_barplot['SD_ABCFA']), data=info_barplot_df, color='mintcream', ax=ax1)
widthbars = [0.8, 0.8, 0.8, 0.8, 0.8,0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
for bar, newwidth in zip(ax2.patches, widthbars):
    x = bar.get_x()
    width = bar.get_width()
    centre = x + width/2.
    bar.set_x(centre - newwidth/2.)
    bar.set_width(newwidth)
ax.set(ylim=(0, 1))

ax.set_ylabel('P. of Responses ', fontsize=15)
ax.set_xlabel('Mean Fundamental Frequency (Hz)', fontsize=15)
ax.set_xticklabels(["144","191 (Control F0)","251","109 ","124 (Control F0)","144 "], fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
ax.set_title('Proportion of Correct Centre Spout Releases'"\n"  'Over Pitch of Target Word, Inter-trial F0 Roving', fontsize=18)
ax3 = ax.twiny()
ax3.set_xlim([0,ax.get_xlim()[1]])
ax3.set_xticks([1.4, 4.15])
ax3.set_xticklabels(['Female', 'Male'], fontsize=12)
#ax[0].set_color('cyan')
fig.tight_layout()
plt.savefig(bin_folder + '\seabornboxplotbehaviouralmeansbypitch_l28.png', bbox_inches='tight')

plt.show()
titanic_dataset = sns.load_dataset("titanic")

sns.barplot(x = "survived", y = "class", data = titanic_dataset)
plt.show()