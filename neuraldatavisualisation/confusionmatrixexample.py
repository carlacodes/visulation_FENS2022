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
import numpy as np
import matplotlib.pyplot as plt
import mat73
import os
import h5py
import numpy as np
import sklearn
from scipy.stats import sem
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

bin_folder='D:/Data/Results/decoderResults/figures/OriginalDataFigures'

list_of_distractors = [20, 57, 32, 56, 42, 2, 5]
meaning_of_distractor=['in contrast to', 'pink noise', 'of science', 'rev. instruments', 'accurate', 'when a', 'craft']
f={}
classifier_data={}
for i in list_of_distractors:
    user_input = 'D:/Data/Results/classifysweepsresults/F1702_Zola_Nellie/Original/training_onroved04092022/l272021/BB2BB3/'
    # directory = os.listdir(user_input)
    print(i)

    # searchstring = ['scoremat*'+str(i)]  # input('What word are you trying to find?')
    # if os.path.isdir(user_input) is False:
    #     print('does not exist')
    #     list_of_distractors.remove(i)
    count=0
    if os.path.isdir(user_input) is True:
        #print('directoryfound')
        directory = os.listdir(user_input)
        searchstring = 'l27' + str(i) # input('What word are you trying to find?')
        for fname in directory:
            if searchstring in fname:
                # Full path
                print('directory loading')
                f[count] = mat73.loadmat(user_input + os.sep + fname)
                items = f[count].items()

                arrays = {}
                for k3, v3 in f[count].items():
                    newarray3 = np.array(v3)
                    #newarrayremove3 = newarray3[0, :]
                    arrays[k3] = newarray3
                classifier_data[i] = arrays
                count=count+1


chansofinterest=[7,8,9, 10, 12, 14,16, 17, 26,  27, 28, 30]
chansofinterest = [x - 1 for x in chansofinterest]
cm_mat_dist_sd_matac=np.empty([])
cm_mat_targ_sd_matac=np.empty([])
cm_mat_targ_mean_ac=np.empty([])
cm_mat_dist_mean_ac=np.empty([])
for k in f:
    print(k)
    cm_mat_dist = np.empty([])
    cm_mat_dist_sd = np.empty([])
    cm_mat_targ = np.empty([])
    cm_mat_targ_sd = np.empty([])
    bigclassmat = np.zeros([1, 12])
    bigstimclasmat = np.zeros([1, 12])
    selectedclassmat=f[k]['classmat']
    selectedclassmat_soundonset=selectedclassmat[:,chansofinterest]
    selectedstimmat=f[k]['stimclassmat']
    selectedstimclassmat_soundonset=selectedstimmat[:, chansofinterest]
    bigclassmat=np.concatenate((bigclassmat, selectedclassmat_soundonset), axis=0)
    bigstimclasmat=np.concatenate((bigstimclasmat, selectedstimclassmat_soundonset), axis=0)

    bigclassmat = np.delete(bigclassmat, 0, 0)
    bigstimclasmat= np.delete(bigstimclasmat, 0, 0)

    # meanbigclassmat=np.mean(bigclassmat, axis=1)
    # meanbigstimclassmat=np.mean(bigstimclasmat, axis=1)

    #y_true=bigclassmat[:,0]
    #run another loop, calculate the confusion matrix score for each site, then take the mean score?
    for ii in np.arange(0,12):
        y_true=bigclassmat[:,ii]
        y_pred=bigstimclasmat[:,ii]
        cm=sklearn.metrics.confusion_matrix(y_true, y_pred)
        distpercent = cm[0, :] / np.sum(cm[0, :])
        distpercent2 = distpercent[0]
        cm_mat_dist = np.append(cm_mat_dist, distpercent2)
        targpercent = cm[1, :] / np.sum(cm[1, :])
        targpercent2 = targpercent[1]

        cm_targ_sd = 1 / np.sqrt(4 * cm[1, :].sum())
        cm_mat_targ = np.append(cm_mat_targ, targpercent2)
        #y_pred=bigstimclasmat[:,0]
    #y_pred=bigstimclasmat.flatten()
    #x = plt.subplot()
    # cm=sklearn.metrics.confusion_matrix(y_true, y_pred)
    # distpercent=cm[0,:]/np.sum(cm[0,:])
    # distpercent2=distpercent[0]
    # cm_mat_dist=np.append(cm_mat_dist, distpercent2)

    cm_mat_targ = np.delete(cm_mat_targ, [0])
    cm_mat_dist = np.delete(cm_mat_dist, [0])
    cm_mat_dist_sd=sem(cm_mat_dist)
    cm_mat_targ_sd=sem(cm_mat_targ)
    #take means and append standard error from mean
    cm_mat_targ_mean=np.mean(cm_mat_targ)
    cm_mat_dist_mean=np.mean(cm_mat_dist)

    cm_mat_targ_sd_matac=np.append(cm_mat_targ_sd_matac, cm_mat_targ_sd)
    cm_mat_dist_sd_matac=np.append(cm_mat_dist_sd_matac, cm_mat_dist_sd)

    cm_mat_targ_mean_ac=np.append(cm_mat_targ_mean_ac, cm_mat_targ_mean)
    cm_mat_dist_mean_ac=np.append(cm_mat_dist_mean_ac, cm_mat_dist_mean)

    cm_mat_targ_sd_matac2=np.delete(cm_mat_targ_sd_matac, 0)
    cm_mat_dist_sd_matac2=np.delete(cm_mat_dist_sd_matac, [0])



    #onfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true', ax=ax, colorbar='True', display_labels=[meaning_of_distractor[k], 'Target'], cmap='Purples')
    #ax=sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Purples');  # annot=True to annotate cells, ftm='g' to disable scientific notation

    # # labels, title and ticks
    # ax.set_xlabel('Predicted labels');
    # ax.set_ylabel('True labels');
    # ax.set_title('Confusion Matrix');
    # ax.xaxis.set_ticklabels([meaning_of_distractor[k], 'instruments']);
    # ax.yaxis.set_ticklabels([meaning_of_distractor[k], 'instruments']);
    #cbar = ax.collections[0].colorbar
    #cbar.set_ticks([0, int(y_true.size/2), y_true.size])
    #cbar.set_ticklabels(['0', '50%', '100%'])

    #cbar = fig.colorbar(ax, ticks=[0, y_true.size/2, y_true.size])
    # #cbar.ax.set_yticklabels(['0', '50%', '100%'])/2  # vertically oriented colorbar
    plt.title('Confusion Matrix for Responses Trained on Original F0, Tested on Roved F0 \n for all Sound Onset Response Channels', fontsize=12)
    plt.savefig(bin_folder + '\confusionmatrixnormalisedtopredictor_dist'+meaning_of_distractor[k]+'.png', dpi=500, bbox_inches='tight')

    plt.show()
barWidth = 0.25
meaning_of_distractor=['in contrast to', 'pink noise', 'of science', 'rev. instruments', 'accurate', 'when a', 'craft']



r1 = np.arange(len(cm_mat_targ))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, cm_mat_targ_mean_ac, yerr=cm_mat_targ_sd_matac, color='darkmagenta', width=barWidth, edgecolor='white', label='Target')
plt.bar(r2, cm_mat_dist_mean_ac, yerr=cm_mat_dist_sd_matac, color='seagreen',width=barWidth, edgecolor='white', label='Distractor')
plt.xticks([r + barWidth for r in range(len(cm_mat_targ))], meaning_of_distractor, fontsize=12)
plt.xticks(rotation = 45)
plt.ylabel('p(CC)', fontsize=12)
plt.title('Proportion of Correct Classifications (CC) Over Distractor ', fontsize=15)
plt.legend(fontsize=12)
plt.savefig(bin_folder + '\confusionmatrixnormalisedtopredictor_dist_BARCHART_sem' + '.png', dpi=500,
            bbox_inches='tight')
plt.show()

