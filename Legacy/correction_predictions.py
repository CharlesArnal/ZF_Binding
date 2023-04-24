import pandas as pd
import os
import numpy as np
from math import comb, floor
import copy
from scipy.special import logsumexp
import collections

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
data_set_preds_file = os.path.join(local_path,"my_exps/results_1.tsv")
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

def partitions_of_n_in_l_summands_w_fixed_positions(n,l):
    if n<0:
        return []
    if l <0 :
        return []
    if l==1:
        return [[n]]
    if n==0:
        return [[0]*l]
    partitions = []
    for i in range(n+1):
        new_partitions = [[i]+partition for partition in partitions_of_n_in_l_summands_w_fixed_positions(n-i,l-1)]
        partitions = partitions+new_partitions
    return partitions

def partitions_of_n_in_l_nonzero_summands_w_fixed_positions(n,l):
    partitions = partitions_of_n_in_l_summands_w_fixed_positions(n-l,l)
    return [[x+1 for x in partition] for partition in partitions]

# useless
def all_0_1_sequences_of_length_n(n):
    if n==0:
        return [[]]
    return [[0]+sequence for sequence in all_0_1_sequences_of_length_n(n-1)]+[[1]+sequence for sequence in all_0_1_sequences_of_length_n(n-1)]


def sequences_from_partitions(partitions):
    sequences = []
    for partition in partitions:
        current_number = 0
        sequence_1 = [] #starts with 0
        sequence_2 = [] #starts with 1
        for integer in partition:
            sequence_1.append([copy.copy(current_number)]*integer)
            sequence_2.append([1-copy.copy(current_number)]*integer)
            current_number = 1-current_number

        sequences.append(sum(sequence_1, start = []))
        sequences.append(sum(sequence_2, start = []))
    return sequences


def generate_all_sequences_with_n_changes(length_sequence, n_changes):
    partitions = partitions_of_n_in_l_nonzero_summands_w_fixed_positions(length_sequence,n_changes+1)
    sequences = sequences_from_partitions(partitions)
    return sequences

def thresholded_proba(pred, threshold = THRESH):
    if pred>= threshold:
        return (pred-threshold)*0.5/(1.0-threshold) + 0.5
    else:
        return pred*0.5/threshold

def log_proba(sequence, preds):
    lp = 0
    if len(sequence)!=len(preds):
        print("Error !!!")
    for index, i in enumerate(sequence):
        modified_proba = thresholded_proba(preds[index])
        if i==1:
            lp+= np.log(modified_proba)
            # TODO changes here
            if len(preds)<10:
                lp+= np.log(1.2)
        else:
            lp+=np.log(1-modified_proba)
            if len(preds) >=10:
                lp+= np.log(1.2)
    return lp



# takes as input the preds for a single protein (and the overall probabilities for each number of changes)
def most_likely_sequence_per_protein(preds,proba_of_n_changes, alpha):
    # TODO : something to avoid preds in {0,1}
    l_protein = len(preds)
    final_sequences = []
    final_scores = []
    #print("\n")
    for n_changes in range(5):
        sequences = generate_all_sequences_with_n_changes(l_protein, n_changes)

        if sequences != []:
            scores = np.array([log_proba(sequence, preds) for sequence in sequences])
            predicted_probability_n_changes= sum([np.exp(score) for score in scores])
            #print(f"A priori probability of {n_changes} changes : {proba_of_n_changes[n_changes]}, predicted_probability : {predicted_probability_n_changes}")
            #renormalized_probability_of_n_changes =  np.log(predicted_probability_n_changes*(1-alpha) + proba_of_n_changes[n_changes]*alpha) # logsumexp([np.log(proba_of_n_changes[n_changes])] + scores )-np.log(2)
            #renormalized_probability_of_n_changes = np.log(proba_of_n_changes[n_changes]**alpha)
            
            # if n_changes == 2:
            #     renormalized_probability_of_n_changes = np.log(1+0.6)
            # elif n_changes == 0:
            #     renormalized_probability_of_n_changes = np.log(1-0.45)
            # else:
            #     renormalized_probability_of_n_changes = 0
            renormalized_probability_of_n_changes = 1
            if l_protein <11:
                if n_changes == 0:
                    renormalized_probability_of_n_changes = 1 + 0.18  + 0.2
                elif n_changes == 1:
                    renormalized_probability_of_n_changes = 1 + 0.5  + 0.2
                elif n_changes == 2:
                    renormalized_probability_of_n_changes = 1 + 0.27   - 0.2
                else: 
                    renormalized_probability_of_n_changes = 1.0 - 0.5
            if l_protein in {11,12,13,14}:
                if n_changes == 1:
                    renormalized_probability_of_n_changes = 1 + 0.4
                elif n_changes == 2:
                    renormalized_probability_of_n_changes = 1 + 0.59 - 0.5
                else: 
                    renormalized_probability_of_n_changes = 1.0 - 0.5
            if l_protein > 14:
                if n_changes == 1:
                    renormalized_probability_of_n_changes = 1 + 0.13 
                elif n_changes == 2:
                    renormalized_probability_of_n_changes = 1 + 0.73 - 0.5
                elif n_changes == 4:
                    renormalized_probability_of_n_changes = 1 + 0.14 - 0.2
                else: 
                    renormalized_probability_of_n_changes = 1.0 - 0.5
            # renormalized_probability_of_n_changes = 1
            # if l_protein<11:
            #     if n_changes > 2:
            #         renormalized_probability_of_n_changes = 0.5
            # if l_protein > 14:
            #     if n_changes not in {1,2,4}:
            #         renormalized_probability_of_n_changes = 0.5
            # else:
            #     if n_changes not in {1,2}:
            #         renormalized_probability_of_n_changes = 0.5
            renormalized_probability_of_n_changes = np.log(renormalized_probability_of_n_changes)
            corrected_scores = scores   + renormalized_probability_of_n_changes #- logsumexp(scores)#+ np.log(proba_of_n_changes[n_changes]) 
            final_sequences += sequences
            final_scores += corrected_scores.tolist()
        # print("\n")
        # print(f"n_changes {n_changes}, logsum {logsumexp(scores)}, log_prob of changes {np.log(proba_of_n_changes[n_changes]) }")
        # print(sequences)
        # print(corrected_scores)
    best_index = np.argmax(np.array(final_scores))
    best_index_initial = np.argmax(np.array(scores))
    #print(f"best index { best_index} best index initial {best_index_initial}")
    best_sequence = final_sequences[best_index]
    return best_sequence


def corrected_preds(preds_per_protein, proba_of_n_changes, alpha):
    corrected_preds = []
    for preds in preds_per_protein:
        corrected_preds += most_likely_sequence_per_protein(preds, proba_of_n_changes, alpha)
    return corrected_preds




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

print(f"Uncorrected percentage of correct answers : {compare_preds_to_labels(preds,labels)}")


# total percentages of ones :
total_percentage_of_ones_true_labels =  len([1 for label in labels if label==1])/float(len(labels))
total_percentage_of_ones_hard_preds =  len([1 for pred in hard_preds if pred==1])/float(len(hard_preds))
print(f"Percentages of ones for true labels {total_percentage_of_ones_true_labels} and hard preds {total_percentage_of_ones_hard_preds}")
# softened_probas_of_n_changes_from_data =


for alpha  in  {0}:#np.linspace(0,2,11):
    my_corrected_preds = corrected_preds(preds_per_protein, proba_of_n_changes_from_data, float(alpha))
    print(f"Corrected percentage of correct answers with correction alpha = {alpha} : {compare_preds_to_labels(my_corrected_preds,labels)}")
    corrected_predicted_proteins = [[] for i in range(num_proteins)]
    for index, protein_id in enumerate(protein_ids):
        corrected_predicted_proteins[protein_id].append(my_corrected_preds[index])
    num_of_proteins_with_n_changes_with_corrected_preds = []
    for i in range(7):
        num_of_proteins_with_n_changes_with_corrected_preds.append(len([protein for protein in corrected_predicted_proteins if num_changes_in_prot(protein)== i ]))
    my_string = ""
    for i in range(6):
        my_string = my_string+f', {i} changes : {num_of_proteins_with_n_changes_with_corrected_preds[i]}'
    print(f"Corrected labels :                   "+my_string[2:])

# print(partitions_of_n_in_l_nonzero_summands_w_fixed_positions(5,3))
# print(all_0_1_sequences_of_length_n(4))
# print(generate_all_sequences_with_n_changes(4,1))
# proba_of_n_changes = [0.5,0,0,0.5,0]
# preds = [0.1,0.9,0.99,0.3,0.25]
#print(most_likely_sequence_per_protein(preds_per_protein[100],proba_of_n_changes_from_data))

"""
Observations : somme donne souvent pas 1 pour les predicted probabilities (important ? Ne change rien pour ce que je fais)
"""

print(sum(abs(np.array(my_corrected_preds)-np.array(hard_preds))))

# for thresh in np.linspace(0.0,1,101):
#     print(f"Uncorrected percentage of correct answers with thresh {thresh}: {compare_preds_to_labels(preds,labels,threshold=thresh)}")




