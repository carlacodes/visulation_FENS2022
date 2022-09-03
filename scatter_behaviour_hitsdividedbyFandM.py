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
fname='scatter_hit_data_dividedbymandF.mat'
info_barplot = rd.loadmat(bin_folder + os.sep + fname)
#matversion_difference=info_barplot['differencemat
mean_hit_data=(info_barplot['combinedmat'])
individual_hit_data=(info_barplot['combinedmatindividual'])

trialnumbers=(info_barplot['combinedMatnonCatch'])
trialnumberscontrol1=np.sum(trialnumbers[0,:])
trialnumberscontrol2=np.sum(trialnumbers[1,:])
intratrialcontrol=mean_hit_data[0,:]
intertrialcontrol=mean_hit_data[1,:]

intratrialcontrol_individual=individual_hit_data[0,:]
intertrialcontrol_individual=individual_hit_data[1,:]

mean_hit_data2=intratrialcontrol*(trialnumberscontrol1/(trialnumberscontrol1+trialnumberscontrol2))+intertrialcontrol*(trialnumberscontrol2/(trialnumberscontrol1+trialnumberscontrol2))
individual_hit_data2=intratrialcontrol_individual*(trialnumberscontrol1/(trialnumberscontrol1+trialnumberscontrol2))+intertrialcontrol_individual*(trialnumberscontrol2/(trialnumberscontrol1+trialnumberscontrol2))

new_mean_hit_Data=np.zeros((3,2))
new_mean_hit_Data[0,:]=mean_hit_data2
new_mean_hit_Data[1,:]=mean_hit_data[1,:]
new_mean_hit_Data[2,:]=mean_hit_data[3,:]

new_individual_hit_Data=np.zeros((4,8))
positive_values = [20, 17.5, 40]
negative_values = [15, 8, 70]
index = ['Female Talker', 'Male Talker']

#'F1702 Zola', 'F1815 Cruella', 'F1803 Tina*', 'F2002 Macaroni*']

df = pd.DataFrame({'Control F0': new_mean_hit_Data[0,:],
                    'Intra-trial Roved F0': mean_hit_data[1,:],'Inter-trial Roved F0': mean_hit_data[3,:] }, index=index)

mac_control_hit=np.zeros((3,2))
mac_control_hit[0,0]=individual_hit_data2[0]
mac_control_hit[0,1]=individual_hit_data2[4]
mac_control_hit[1,0]=individual_hit_data[1,0]
mac_control_hit[1,1]=individual_hit_data[1,4]
mac_control_hit[2,0]=individual_hit_data[3,0]
mac_control_hit[2,1]=individual_hit_data[3,4]

tina_control_hit=np.zeros((3,2))
tina_control_hit[0,0]=individual_hit_data2[1]
tina_control_hit[0,1]=individual_hit_data2[5]
tina_control_hit[1,0]=individual_hit_data[1,1]
tina_control_hit[1,1]=individual_hit_data[1,5]
tina_control_hit[2,0]=individual_hit_data[3,1]
tina_control_hit[2,1]=individual_hit_data[3,5]



cruella_control_hit=np.zeros((3,2))
cruella_control_hit[0,0]=individual_hit_data2[2]
cruella_control_hit[0,1]=individual_hit_data2[6]
cruella_control_hit[1,0]=individual_hit_data[1,2]
cruella_control_hit[1,1]=individual_hit_data[1,6]
cruella_control_hit[2,0]=individual_hit_data[3,2]
cruella_control_hit[2,1]=individual_hit_data[3,6]


zola_control_hit=np.zeros((3,2))
zola_control_hit[0,0]=individual_hit_data2[3]
zola_control_hit[0,1]=individual_hit_data2[7]
zola_control_hit[1,0]=individual_hit_data[1,3]
zola_control_hit[1,1]=individual_hit_data[1,7]
zola_control_hit[2,0]=individual_hit_data[3,3]
zola_control_hit[2,1]=individual_hit_data[3,7]



df_individual_macaroni = pd.DataFrame({'Control F0': mac_control_hit[0,:],
                    'Intra-trial Roved F0': mac_control_hit[1,:],'Inter-trial Roved F0': mac_control_hit[2,:], }, index=index)

df_individual_tina = pd.DataFrame({'Control F0': tina_control_hit[0,:],
                    'Intra-trial Roved F0': tina_control_hit[1,:],'Inter-trial Roved F0': tina_control_hit[2,:], }, index=index)

df_individual_cruella = pd.DataFrame({'Control F0': cruella_control_hit[0,:],
                    'Intra-trial Roved F0': cruella_control_hit[1,:],'Inter-trial Roved F0': cruella_control_hit[2,:], }, index=index)

df_individual_zola = pd.DataFrame({'Control F0': zola_control_hit[0,:],
                    'Intra-trial Roved F0': zola_control_hit[1,:],'Inter-trial Roved F0': zola_control_hit[2,:], }, index=index)

ax = df.plot.bar(rot=0)
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
# sns.scatterplot(data=df_individual_macaroni, legend=False, zorder=10, x_jitter=5,  markers=markers_mac, label='F2002')
#
# sns.scatterplot(data=df_individual_tina, legend=False, zorder=10, x_jitter=5,  markers=markers_tina, label='F1803')
# sns.scatterplot(data=df_individual_cruella, legend=False, zorder=10, x_jitter=5,  markers=markers_cruella, label='F1815')
# sns.scatterplot(data=df_individual_zola, legend=False, zorder=10, x_jitter=5,  markers=markers_zola, label='F1702')

plt.scatter([-0.1],mac_control_hit[0,0],c='blue', label='F2002')
plt.scatter([0.01],mac_control_hit[1,0], c='orange')
plt.scatter([0.1],mac_control_hit[2,0], c='green')

plt.scatter([1],mac_control_hit[0,1],c='blue')
plt.scatter([1.1],mac_control_hit[1,1], c='orange')
plt.scatter([1.2],mac_control_hit[2,1], c='green')

plt.scatter([-0.15],tina_control_hit[0,0],c='blue', marker='X', label='F1803')
plt.scatter([0.05],tina_control_hit[1,0], c='orange', marker='X')
plt.scatter([0.15],tina_control_hit[2,0], c='green', marker='X')

plt.scatter([0.8],tina_control_hit[0,1],c='blue', marker='X')
plt.scatter([0.9],tina_control_hit[1,1], c='orange', marker='X')
plt.scatter([1.1],tina_control_hit[2,1], c='green', marker='X')

plt.scatter([-0.16],cruella_control_hit[0,0],c='blue', marker='^', label='F1815')
plt.scatter([0.07],cruella_control_hit[1,0], c='orange',marker='^')
plt.scatter([0.13],cruella_control_hit[2,0], c='green', marker='^')

plt.scatter([0.85],cruella_control_hit[0,1],c='blue', marker='^')
plt.scatter([0.95],cruella_control_hit[1,1], c='orange', marker='^')
plt.scatter([1.15],cruella_control_hit[2,1], c='green', marker='^')

plt.scatter([-0.14],zola_control_hit[0,0],c='blue', marker='s', label='F1702')
plt.scatter([0.03],zola_control_hit[1,0], c='orange',marker='s',)
plt.scatter([0.13],zola_control_hit[2,0], c='green', marker='s',)

plt.scatter([0.85],zola_control_hit[0,1],c='blue', marker='s',)
plt.scatter([0.95],zola_control_hit[1,1], c='orange', marker='s',)
plt.scatter([1.15],zola_control_hit[2,1], c='green', marker='s',)


tips = sns.load_dataset("tips")

plt.ylabel('Proportion of Hits')
plt.xlabel('Talker Sex')
ax.set_ylim([0, 1])
plt.legend(fontsize=8)
plt.title('Proportion of Hits Over Talker Type')
plt.show()