import seaborn as sns
import numpy as np
import mat73
import os
import scipy.io as rd
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import pandas as pd
import matplotlib.pyplot as plt

bin_folder='D:/Data/Results/L28general/nopitchshift'
fname='scatter_percentcorrect_data_dividedbymandFonlyzoladist.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)
#matversion_difference=info_barplot['differencemat
mean_hit_data=(info_barplot['combinedmat'])
individual_hit_data=(info_barplot['combinedmatindividual'])
combinedmatSDindividual=(info_barplot['combinedmatSDindividual'])

trialnumbers1=(info_barplot['combinedMatCatch'])
trialnumbers2=(info_barplot['combinedMatnonCatch'])
trialnumbers=trialnumbers1+trialnumbers2

trialnumberscontrol1=np.sum(trialnumbers[0,:])
trialnumberscontrol2=np.sum(trialnumbers[1,:])
intratrialcontrol=np.concatenate((mean_hit_data[:,2],mean_hit_data[:,3]))
#intertrialcontrol=np.concatenate((mean_hit_data[:,6],mean_hit_data[:,7]))

intratrialcontrol_individual=np.concatenate((individual_hit_data[:,2], individual_hit_data[:,3]))
#flattening so now all the columns represent individual ferrets
#intertrialcontrol_individual=np.concatenate((individual_hit_data[:,6], individual_hit_data[:,7]))

mean_hit_data2= intratrialcontrol#intratrialcontrol*(trialnumberscontrol1/(trialnumberscontrol1+trialnumberscontrol2))+intertrialcontrol*(trialnumberscontrol2/(trialnumberscontrol1+trialnumberscontrol2))
individual_hit_data2=intratrialcontrol_individual#intratrialcontrol_individual*(trialnumberscontrol1/(trialnumberscontrol1+trialnumberscontrol2))+intertrialcontrol_individual*(trialnumberscontrol2/(trialnumberscontrol1+trialnumberscontrol2))

new_mean_hit_Data=np.zeros((3,2))
new_mean_hit_Data[0,:]=mean_hit_data2
##row 1 = intra trial roving, each column representING f or M
new_mean_hit_Data[1,:]=np.concatenate((mean_hit_data[:,0],mean_hit_data[:,1]))
#new_mean_hit_Data[2,:]=np.concatenate((mean_hit_data[:,4],mean_hit_data[:,5]))

new_individual_hit_Data=np.zeros((4,8))
meaning_of_distractor=['in contrast to', 'pink noise', 'of science', 'rev. instruments', 'accurate', 'when a', 'craft']

index = ['Female', 'Male']
#
# #'F1702 Zola', 'F1815 Cruella', 'F1803 Tina*', 'F2002 Macaroni*']
#
# df = pd.DataFrame({'Control': new_mean_hit_Data[0,:],
#                     'Intra': new_mean_hit_Data[1,:],'Inter': new_mean_hit_Data[2,:] }, index=index)
# #macaroni is the first row for individual_hit_data
# mac_control_hit=np.zeros((3,2))
# mac_control_hit[0,0]=individual_hit_data2[0]
# mac_control_hit[0,1]=individual_hit_data2[4]
# #intra trial roving should be the second row
# mac_control_hit[1,0]=individual_hit_data[0,0]
# mac_control_hit[1,1]=individual_hit_data[0,1]
# #intra trial roving should be the third row
# mac_control_hit[2,0]=individual_hit_data[0,4]
# mac_control_hit[2,1]=individual_hit_data[0,5]
#
# tina_control_hit=np.zeros((3,2))
# tina_control_hit[0,0]=individual_hit_data2[1]
# tina_control_hit[0,1]=individual_hit_data2[5]
# tina_control_hit[1,0]=individual_hit_data[1,0]
# tina_control_hit[1,1]=individual_hit_data[1,1]
# tina_control_hit[2,0]=individual_hit_data[1,4]
# tina_control_hit[2,1]=individual_hit_data[1,5]
#
#
#
# cruella_control_hit=np.zeros((3,2))
# cruella_control_hit[0,0]=individual_hit_data2[2]
# cruella_control_hit[0,1]=individual_hit_data2[6]
# cruella_control_hit[1,0]=individual_hit_data[2,0]
# cruella_control_hit[1,1]=individual_hit_data[2,1]
# cruella_control_hit[2,0]=individual_hit_data[2,4]
# cruella_control_hit[2,1]=individual_hit_data[2,5]
#
#
# zola_control_hit=np.zeros((3,2))
# zola_control_hit[0,0]=individual_hit_data2[3]
# zola_control_hit[0,1]=individual_hit_data2[7]
# zola_control_hit[1,0]=individual_hit_data[3,0]
# zola_control_hit[1,1]=individual_hit_data[3,1]
# zola_control_hit[2,0]=individual_hit_data[3,4]
# zola_control_hit[2,1]=individual_hit_data[3,5]
#
#
#
# df_individual_macaroni = pd.DataFrame({'Control F0': mac_control_hit[0,:],
#                     'Intra': mac_control_hit[1,:],'Inter': mac_control_hit[2,:], }, index=index)
#
# df_individual_tina = pd.DataFrame({'Control F0': tina_control_hit[0,:],
#                     'Intra=': tina_control_hit[1,:],'Inter': tina_control_hit[2,:], }, index=index)
#
# df_individual_cruella = pd.DataFrame({'Control F0': cruella_control_hit[0,:],
#                     'Intra': cruella_control_hit[1,:],'Inter': cruella_control_hit[2,:], }, index=index)
#
# df_individual_zola = pd.DataFrame({'Control F0': np.concatenate((individual_hit_data[:,2], individual_hit_data[:,3]), axis=0),
#                     'Intra': np.concatenate((individual_hit_data[:,0], individual_hit_data[:,3]), axis=0)}, index=index)

# set width of bars
barWidth = 0.25

# set heights of bars
controlintraF =individual_hit_data[:,2]
controlintraFSD=combinedmatSDindividual[:,2]
controlintraM =individual_hit_data[:,3]
controlintraMSD=combinedmatSDindividual[:,3]

intraF =individual_hit_data[:,0]
intraFSD=combinedmatSDindividual[:,0]
intraM=individual_hit_data[:,1]
intraMSD=combinedmatSDindividual[:,1]

bars2 = [28, 6, 16, 5, 10]
bars3 = [29, 3, 24, 25, 17]

# Set position of bar on X axis
r1 = np.arange(len(controlintraF))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, controlintraF, color='#0343DF', yerr=controlintraFSD, width=barWidth, edgecolor='white', label='Control')
plt.bar(r2, intraF, color='#f97306', yerr=intraFSD, width=barWidth, edgecolor='white', label='Intra')

#plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

# Add xticks on the middle of the group bars
#plt.xlabel('distractor', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(controlintraF))], meaning_of_distractor)
plt.xticks(rotation = 45)
# Create legend & Show graphic
plt.ylabel('p(CR)', fontsize=15)

#plt.xlabel('Talker', fontsize=15)

plt.legend(fontsize=8)
plt.title('Proportion of Correct Responses (CR) \n Over Distractor (Female Talker)', fontsize=18)
plt.ylim([0,1])
plt.savefig(bin_folder + '\proportionofpercentcorrectrejectionsoverdisttypezola.png', dpi=500,
            bbox_inches='tight')
plt.show()


plt.bar(r1, controlintraM, color='#0343DF',yerr=controlintraMSD, width=barWidth, edgecolor='white', label='Control')
plt.bar(r2, intraM, color='#f97306', yerr=intraMSD, width=barWidth, edgecolor='white', label='Intra')
#plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

# Add xticks on the middle of the group bars
#plt.xlabel('distractor', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(controlintraF))], meaning_of_distractor)
plt.xticks(rotation = 45)
# Create legend & Show graphic
plt.ylabel('p(CR)', fontsize=15)

#plt.xlabel('Talker', fontsize=15)

plt.legend(fontsize=8)
plt.title('Proportion of Correct Responses (CR) \n Over Distractor (Male Talker)', fontsize=18)
plt.ylim([0,1])

plt.savefig(bin_folder + '\percentcorrectrejectionsoverdisttypezolaMtalker.png', dpi=500,
            bbox_inches='tight')
plt.show()

def jitter_dots(dots):
    offsets = dots.get_offsets()
    jittered_offsets = offsets
    # only jitter in the x-direction
    jittered_offsets[:, 0] += np.random.uniform(-0.3, 0.3, offsets.shape[0])
    dots.set_offsets(jittered_offsets)
#sns.pointplot(data=df_individual_macaroni,linestyles='', markers='o', ax=ax)
markers_mac = {"Control F0": "X", "Intra-trial Roved F0": "X", 'Inter-trial Roved F0': 'X'}
markers_tina = {"Control F0": "o", "Intra-trial Roved F0": "o", 'Inter-trial Roved F0': 'o'}
markers_cruella = {"Control F0": "^", "Intra-trial Roved F0": "^", 'Inter-trial Roved F0': '^'}
markers_zola = {"Control F0": "2", "Intra-trial Roved F0": "2", 'Inter-trial Roved F0': '2'}
def jitter(values,j):
    return values + np.random.normal(j,0.1,values.shape)
# tuples=('Control F0', 'Intra-trial Roved F0', 'Inter-trial Roved F0')
# index2 = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

#macplot=sns.stripplot(data=df_individual_macaroni.T,jitter=0.01, marker='X', label='F2002')
# sns.scatterplot(data=df_individual_macaroni, legend=percentcorrectlse, zorder=10, x_jitter=5,  markers=markers_mac, label='F2002')
#
# sns.scatterplot(data=df_individual_tina, legend=percentcorrectlse, zorder=10, x_jitter=5,  markers=markers_tina, label='F1803')
# sns.scatterplot(data=df_individual_cruella, legend=percentcorrectlse, zorder=10, x_jitter=5,  markers=markers_cruella, label='F1815')
# sns.scatterplot(data=df_individual_zola, legend=percentcorrectlse, zorder=10, x_jitter=5,  markers=markers_zola, label='F1702')

# plt.scatter([-0.1],mac_control_hit[0,0],c='blue', label='F2002')
# plt.scatter([0.01],mac_control_hit[1,0], c='orange')
# plt.scatter([0.13],mac_control_hit[2,0], c='green')
#
# plt.scatter([0.8],mac_control_hit[0,1],c='blue')
# plt.scatter([0.99],mac_control_hit[1,1], c='orange')
# plt.scatter([1.2],mac_control_hit[2,1], c='green')
#
# plt.scatter([-0.2],tina_control_hit[0,0],c='blue', marker='X', label='F1803')
# plt.scatter([0.05],tina_control_hit[1,0], c='orange', marker='X')
# plt.scatter([0.12],tina_control_hit[2,0], c='green', marker='X')
#
# plt.scatter([0.86],tina_control_hit[0,1],c='blue', marker='X')
# plt.scatter([1.02],tina_control_hit[1,1], c='orange', marker='X')
# plt.scatter([1.1],tina_control_hit[2,1], c='green', marker='X')
#
# plt.scatter([-0.16],cruella_control_hit[0,0],c='blue', marker='^', label='F1815')
# plt.scatter([0.07],cruella_control_hit[1,0], c='orange',marker='^')
# plt.scatter([0.17],cruella_control_hit[2,0], c='green', marker='^')
#
# plt.scatter([0.85],cruella_control_hit[0,1],c='blue', marker='^')
# plt.scatter([1.05],cruella_control_hit[1,1], c='orange', marker='^')
# plt.scatter([1.15],cruella_control_hit[2,1], c='green', marker='^')
#
# plt.scatter([-0.14],zola_control_hit[0,0],c='blue', marker='s', label='F1702')
# plt.scatter([0.03],zola_control_hit[1,0], c='orange',marker='s',)
# plt.scatter([0.2],zola_control_hit[2,0], c='green', marker='s',)
#
# plt.scatter([0.85],zola_control_hit[0,1],c='blue', marker='s',)
# plt.scatter([0.95],zola_control_hit[1,1], c='orange', marker='s',)
# plt.scatter([1.22],zola_control_hit[2,1], c='green', marker='s',)


tips = sns.load_dataset("tips")



