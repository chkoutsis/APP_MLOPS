'''
N -> Press
W -> Release

HT = W(i-1) - N(i-1)
PPT = Ni - N(i-1)
RRT = Wi - W(i-1)
RPT = Ni - W(i-1)
'''

import os
import pandas as pd
import numpy as np

path = os.getcwd()
data = pd.read_csv(path + '\Train_keystroke.csv')

# creating a dataframe
df = pd.DataFrame() 

#  generate HT, PPT, RRT, RPT for each two consecutive keys.
for _, row in data.iterrows():

    list_ht = [row[2] - row[1]]
    list_ppt = []
    list_rrt = []
    list_rpt = []

    for i in range(4, len(row), 2):
        ht = row[i] - row[i-1]
        ppt = row[i-1] - row[i-3]
        rrt = row[i] - row[i-2]
        rpt = row[i-1] - row[i-2]

        list_ht.append(ht)
        list_ppt.append(ppt)
        list_rrt.append(rrt)
        list_rpt.append(rpt)


    mean_ht = np.mean(list_ht)
    mean_ppt = np.mean(list_ppt)
    mean_rrt = np.mean(list_rrt)
    mean_rpt = np.mean(list_rpt)

    std_ht = np.std(list_ht)
    std_ppt = np.std(list_ppt)
    std_rrt = np.std(list_rrt)
    std_rpt = np.std(list_rpt)

    data = {'mean_ht': mean_ht, 'std_ht': std_ht,
            'mean_ppt': mean_ppt, 'std_ppt': std_ppt,
            'mean_rrt': mean_rrt, 'std_rrt': std_rrt,
            'mean_rpt': mean_rpt, 'std_rpt': std_rpt,
            'UserID': row[0]}
    new_df =  pd.DataFrame(pd.Series(data), index=None).T
    new_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, new_df], ignore_index=True,  axis=0)
print(df)

df.to_csv(path + '//aaaa.csv', index=False)


