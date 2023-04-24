import pandas as pd
import os
import numpy as np
from math import comb, floor
import copy
from scipy.special import logsumexp
import collections

from sklearn.metrics import confusion_matrix

THRESH = 0.825

def num_changes_in_prot(protein):
    n = 0
    for i in range(len(protein)-1):
        if protein[i]!=protein[i+1]:
            n+=1
    return n

def proba_n_changes(protein, absolute_proba_change,n):
    l = len(protein)-1
    if n <0 or n > l:
        return 0
    else:
        return absolute_proba_change**n*(1-absolute_proba_change)**(l-n)*comb(l,n)



local_path = '/home/charles/Desktop/Dan_project/DeepZF-main'
data_set_file = os.path.join(local_path,"Data/BindZFpredictor/0_zf_0_b.csv")
data_set = pd.read_csv(data_set_file).dropna()
data_set_preds_file = os.path.join(local_path,"my_exps/1st_exp_github.tsv")
data_set_preds = np.loadtxt(data_set_preds_file,  delimiter='\t')

labels = data_set["label"].tolist()
protein_ids = data_set["groups"].tolist()
num_proteins = max(protein_ids)+1
print(f"Total number of proteins = {num_proteins}")
proteins = [[] for i in range(num_proteins)]
for index, protein_id in enumerate(protein_ids):
    proteins[protein_id].append(labels[index])

preds = data_set_preds.tolist()
preds_per_protein= [[] for i in range(num_proteins)]
for index, protein_id in enumerate(protein_ids):
    preds_per_protein[protein_id].append(preds[index])

def compare_preds_to_labels(preds,labels,threshold = THRESH):
    correct = 0.0
    for index, pred in enumerate(preds):
        if pred >=threshold:
            if labels[index]==1:
                correct+=1
        else:
            if labels[index]==0:
                correct+=1
    return correct/len(preds)




# probability of change
n_changes = 0.0
for i in range(len(labels)-1):
    if labels[i]!=labels[i+1]:
        n_changes +=1.0
p_changes = n_changes/float(len(labels)-1)

print(f"# labels : {len(labels)}, # 1 : {len([i for i in labels if i==1])}, # 0 : {len([i for i in labels if i==0])}, p_change : {p_changes}")

# Compute hard labels from soft preds :
hard_preds = [(1 if pred >=THRESH else 0) for pred in preds  ]
predicted_proteins = [[] for i in range(num_proteins)]
for index, protein_id in enumerate(protein_ids):
    predicted_proteins[protein_id].append(hard_preds[index])

# Compute number of changes per proteins
average_len_protein = np.mean(np.array([len(protein) for protein in proteins]))
num_of_proteins_with_n_changes = []
for i in range(7):
    num_of_proteins_with_n_changes.append(len([protein for protein in proteins if num_changes_in_prot(protein)== i ]))

proba_of_n_changes_from_data = np.array(num_of_proteins_with_n_changes)/float(num_proteins)

print(f"Average length of a protein: {average_len_protein}")

# Print number of proteins with n changes
my_string = ""
for i in range(6):
    my_string = my_string+f', {i} changes : {num_of_proteins_with_n_changes[i]}'
print(f"True labels :                        "+my_string[2:])

# Compute number of changes per proteins using hard preds
num_of_proteins_with_n_changes_with_hard_labels = []
for i in range(7):
    num_of_proteins_with_n_changes_with_hard_labels.append(len([protein for protein in predicted_proteins if num_changes_in_prot(protein)== i ]))

# Print number of proteins with n changes if using hard preds
my_string = ""
for i in range(6):
    my_string = my_string+f', {i} changes : {num_of_proteins_with_n_changes_with_hard_labels[i]}'
print(f"Predicted labels :                   "+my_string[2:])
        
# Expected number of proteins with n changes if independent
my_string = ""
for i in range(6):
    expected_number_of_configurations = sum([proba_n_changes(protein, p_changes, i) for protein in proteins])
    my_string = my_string+f', {i} changes : {floor(expected_number_of_configurations)}'
print(f"Expected numbers of configurations : "+my_string[2:])

# Compute the number of changes as a function of length of protein (and vice versa)
changes_per_length = dict()
for protein in proteins:
    if len(protein) in changes_per_length.keys():
        changes_per_length[len(protein)][num_changes_in_prot(protein)]+=1
    else:
        changes_per_length[len(protein)] = [0]*6
        changes_per_length[len(protein)][num_changes_in_prot(protein)] =1

mean_probas_0 = np.sum(np.array([changes_per_length[i] for i in range(11) if i in changes_per_length.keys()]), axis=0)
mean_probas_1 = np.sum(np.array([changes_per_length[i] for i in {11,12,13,14}  if i in changes_per_length.keys()]), axis=0) 
mean_probas_2 = np.sum(np.array([changes_per_length[i] for i in range(15,21)  if i in changes_per_length.keys()]), axis=0) 

print(f"Mean probas : 0 : {mean_probas_0/sum(mean_probas_0)}, 1 : {mean_probas_1/sum(mean_probas_1)}, 2 : {mean_probas_2/sum(mean_probas_2)}")

lengths_per_changes = dict()
for protein in proteins:
    if num_changes_in_prot(protein) in lengths_per_changes.keys():
        lengths_per_changes[num_changes_in_prot(protein)][len(protein)]+=1
    else:
        lengths_per_changes[num_changes_in_prot(protein)] = [0]*21
        lengths_per_changes[num_changes_in_prot(protein)][len(protein)] =1

print(f"Changes per length {changes_per_length}")
print(f"Lengths per change {lengths_per_changes}")


percentage_of_ones_per_length = collections.OrderedDict()
for length in range(3,21):
    percentage_of_ones_per_length[length]  = []
for protein in proteins :
    if len(protein) in percentage_of_ones_per_length.keys():
        percentage_of_ones_per_length[len(protein)].append(sum(protein))
    else:
        percentage_of_ones_per_length[len(protein)] = []
        percentage_of_ones_per_length[len(protein)].append(sum(protein))
for key in percentage_of_ones_per_length.keys():
    percentage_of_ones_per_length[key] = np.mean(percentage_of_ones_per_length[key])/float(key)

print(f"Percentage of ones per length : {percentage_of_ones_per_length}")

print(f"Percentage of correct answers : {compare_preds_to_labels(preds,labels)}")


# total percentages of ones :
total_percentage_of_ones_true_labels =  len([1 for label in labels if label==1])/float(len(labels))
total_percentage_of_ones_hard_preds =  len([1 for pred in hard_preds if pred==1])/float(len(hard_preds))
print(f"Percentages of ones for true labels {total_percentage_of_ones_true_labels} and hard preds {total_percentage_of_ones_hard_preds}")
# softened_probas_of_n_changes_from_data =


# Confusion matrix as a function of threshold :
threshold_confusion_matrix  = 0.53

hard_preds_confusion_matrix = [(1 if pred >= threshold_confusion_matrix else 0) for pred in preds  ]
my_confusion_matrix = confusion_matrix(labels, hard_preds_confusion_matrix)
print(my_confusion_matrix)