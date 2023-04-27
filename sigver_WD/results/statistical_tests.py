#### statistical tests
    
import pickle
import numpy as np
from scipy import stats
import os
os.chdir('/home/ellab_dl/Desktop/GitTest/sigver_WD')

# model 1
file_to_read1 = open("results/SigNet_N12_v952_cedar.pkl", "rb")
results1 = pickle.load(file_to_read1)
eer_u_list1 = []
for k in range(10):
    this_eer_u1 = results1[k]['all_metrics']['EER_userthresholds']
    eer_u_list1.append(this_eer_u1)

# model 2  
file_to_read2 = open("results/ResNet18_CL_KD_GEOM_BC_v952_cedar.pkl", "rb")
results2 = pickle.load(file_to_read2)
eer_u_list2 = []
for k in range(10):
    this_eer_u2 = results2[k]['all_metrics']['EER_userthresholds']
    eer_u_list2.append(this_eer_u2)
    
# Wilcoxon paired signed-rank tests with a 5% level of signicance.
# In order to be significant at the 5% level, the test should have P-value smaller than 0.05.
stat, p = stats.wilcoxon(eer_u_list1, eer_u_list2)
p

# Friedman Test
# At α = 0.05, p<a leads to statistically significant differences
'''stats.friedmanchisquare(eer_u_list1, eer_u_list2)'''
'''import pingouin as pg
import pandas as pd
df = pd.DataFrame({
        'model1': [eer_u_list1], 
        'model2': [eer_u_list2]})
pg.friedman(df, method="f") '''
import scikit_posthocs as sp
# rows are blocks and columns are groups
eer_u_arr1 = np.array(eer_u_list1)
eer_u_arr2 = np.array(eer_u_list2)
df = np.vstack((eer_u_arr1, eer_u_arr2))
df = df.transpose()
sp.posthoc_nemenyi_friedman(df)

