import pandas as pd
import os
import numpy as np
from math import comb, floor
import copy
from scipy.special import logsumexp
import collections

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def true_positive_rate(preds, labels, threshold):
    positive_decisions_and_binding = 0.0
    num_bindings = 0.0
    for index, pred in enumerate(preds):
        if labels[index] == 1:
            num_bindings +=1
            if pred >=threshold:
                positive_decisions_and_binding += 1
    return positive_decisions_and_binding/num_bindings

def false_positive_rate(preds, labels, threshold):
    positive_decisions_and_not_binding = 0.0
    num_not_bindings = 0.0
    for index, pred in enumerate(preds):
        if labels[index] == 0:
            num_not_bindings += 1
            if pred >=threshold:
                positive_decisions_and_not_binding += 1
    return positive_decisions_and_not_binding/num_not_bindings


def percentage_of_binding_above_thresh(preds, labels, thresh):
    above_thresh_and_binding = 0.0
    above_thresh = 0.0
    for index, pred in enumerate(preds):
        if pred >=thresh:
            above_thresh += 1
            if labels[index] == 1:
                above_thresh_and_binding += 1
    return (above_thresh_and_binding/above_thresh if above_thresh != 0.0 else 1)

def percentage_of_non_binding_below_thresh(preds,labels, thresh):
    below_thresh_and_non_binding = 0.0
    below_thresh = 0.0
    for index, pred in enumerate(preds):
        if pred <thresh:
            below_thresh += 1
            if labels[index] == 0:
                below_thresh_and_non_binding += 1
    return (below_thresh_and_non_binding/below_thresh if below_thresh != 0.0 else 1)



local_path = '/home/charles/Desktop/ZF_Binding/DeepZF-main'
data_set_file = os.path.join(local_path,"Data/BindZFpredictor/40_zf_40_b.csv")
data_set = pd.read_csv(data_set_file).dropna()
data_set_preds_file = os.path.join(local_path,"Zkscan3_exps/binding_preds_40.tsv")
data_set_preds = np.loadtxt(data_set_preds_file,  delimiter='\t')
preds = data_set_preds.tolist()

labels = data_set["label"].tolist()
protein_ids = data_set["groups"].tolist()
num_proteins = max(protein_ids)+1
print(f"Total number of proteins = {num_proteins}")
print(f"Total number of ZFs {len(labels)}")
assert len(labels) == len(data_set_preds)





tpr = [true_positive_rate(preds, labels, thresh ) for thresh in np.linspace(0,1,100)]
fpr = [false_positive_rate(preds, labels, thresh ) for thresh in np.linspace(0,1,100)]

# print(f"Approximated area under curve = {np.mean(tpr)}")
# plt.plot(fpr,tpr)
# plt.show()

threshs = np.linspace(0,1,100)
pbat = [percentage_of_binding_above_thresh(preds, labels, thresh )*100 for thresh in threshs]
pnbbt = [percentage_of_non_binding_below_thresh(preds, labels, thresh )*100 for thresh in threshs]


fig1 = plt.figure()
plt.plot(threshs, pbat )
plt.xlabel("Threshold")
plt.ylabel("% of binding ZFs above the threshold")
plt.xticks(ticks = np.linspace(0,1,11))
plt.yticks(ticks = np.linspace(40,100,13))
plt.grid(axis="y")

fig2 = plt.figure()
plt.plot(threshs, pnbbt )
plt.xlabel("Threshold")
plt.ylabel("% of non-binding ZFs below the threshold")
plt.xticks(ticks = np.linspace(0,1,11))
plt.yticks(ticks = np.linspace(40,100,13))
plt.grid(axis="y")

plt.show()

print(f"Percentage of binding above {0.8} : {percentage_of_binding_above_thresh(preds, labels, 0.8)}")
print(f"Percentage of  non-binding below {0.8} : {percentage_of_non_binding_below_thresh(preds, labels, 0.8)}")

# proteins = [[] for i in range(num_proteins)]
# for index, protein_id in enumerate(protein_ids):
#     proteins[protein_id].append(labels[index])

# preds = data_set_preds.tolist()
# preds_per_protein= [[] for i in range(num_proteins)]
# for index, protein_id in enumerate(protein_ids):
#     preds_per_protein[protein_id].append(preds[index])
