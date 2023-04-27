import pickle

path_results = '/home/ellab_dl/Desktop/GitTest/sigver_WD/results/'
file_to_read = open(path_results + "ResNet18_CL_KD_GEOM_BC_v952_cedar.pkl", "rb")

loaded_dictionary = pickle.load(file_to_read)

'''
k = 0 # k-repetition (10-Folds, 0-9 repetitions)
writer = 0 # writer ID
loaded_dictionary[k]['all_metrics']['FRR']
loaded_dictionary[k]['predictions']['skilledPreds'][writer]
'''

import numpy as np
eer_u_list = []
eer_list = []
all_results = []
folds = 10
for k in range(folds):
    this_eer_u, this_eer = loaded_dictionary[k]['all_metrics']['EER_userthresholds'], loaded_dictionary[k]['all_metrics']['EER']
    all_results.append(loaded_dictionary)
    eer_u_list.append(this_eer_u)
    eer_list.append(this_eer)

print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))

'''
###################################################################3
{'all_metrics': {'FRR': FRR,
                 'FAR_random': FAR_random,
                 'FAR_skilled': FAR_skilled,
                 'mean_AUC': meanAUC,
                 'EER': EER,
                 'EER_userthresholds': EER_userthresholds,
                 'auc_list': aucs,
                 'global_threshold': global_threshold},

'predictions': {'genuinePreds': genuinePreds,
                'randomPreds': randomPreds,
                'skilledPreds': skilledPreds}}
'''

