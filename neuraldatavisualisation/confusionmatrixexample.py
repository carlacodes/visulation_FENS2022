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


list_of_distractors = [20, 57, 32, 56, 42, 2, 5]
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
    count=0;
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

bigclassmat=np.array([])
bigstimclasmat=np.array([])
chansofinterest=[7,8,9, 10, 12, 14,16, 17, 26,  27, 28, 30]
chansofinterest = [x - 1 for x in chansofinterest]
for k in f:
    print(k)
    selectedclassmat=f[k]['classmat']
    selectedclassmat_soundonset=selectedclassmat[:,chansofinterest]
    selectedstimmat=f[k]['stimclassmat']
    selectedstimclassmat_soundonset=selectedstimmat[:, chansofinterest]
