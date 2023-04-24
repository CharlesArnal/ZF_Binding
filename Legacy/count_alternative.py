import pandas as pd
import os
import numpy as np
from math import comb, floor

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

labels = data_set["label"].tolist()
protein_ids = data_set["groups"].tolist()
num_proteins = max(protein_ids)+1
print(f"Total number of proteins = {num_proteins}")
proteins = [[] for i in range(num_proteins)]
for index, protein_id in enumerate(protein_ids):
    proteins[protein_id].append(labels[index])



# probability of change
n_changes = 0.0
for i in range(len(labels)-1):
    if labels[i]!=labels[i+1]:
        n_changes +=1.0
p_changes = n_changes/float(len(labels)-1)

print(f"# labels : {len(labels)}, # 1 : {len([i for i in labels if i==1])}, # 0 : {len([i for i in labels if i==0])}, p_change : {p_changes}")

# number of changes per proteins
average_len_protein = np.mean(np.array([len(protein) for protein in proteins]))
change_0 = len([protein for protein in proteins if num_changes_in_prot(protein)== 0 ])
change_1 = len([protein for protein in proteins if num_changes_in_prot(protein)== 1 ])
change_2 = len([protein for protein in proteins if num_changes_in_prot(protein)== 2 ])
change_3 = len([protein for protein in proteins if num_changes_in_prot(protein)== 3 ])
change_4 = len([protein for protein in proteins if num_changes_in_prot(protein)== 4 ])
change_mt4 = len([protein for protein in proteins if num_changes_in_prot(protein)> 4 ])

print(f"Average length : {average_len_protein}, 0 changes : {change_0}, 1 changes : {change_1}, 2 changes : {change_2}, 3 changes : {change_3}, 4 changes : {change_4}, > 4 changes : {change_mt4}")

my_string = ""
for i in range(6):
    expected_number_of_configurations = sum([proba_n_changes(protein, p_changes, i) for protein in proteins])
    my_string = my_string+f', {i} changes : {floor(expected_number_of_configurations)}'
print(f"Expected numbers of configurations : "+my_string[2:])

