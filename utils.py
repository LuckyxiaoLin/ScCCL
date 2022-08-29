import os
import numpy as np


def readCSV(datasets):
    nameT = ['klein', 'pollen']
    fp = os.getcwd()
    file = open(fp + '/../data/' + datasets + '.csv')
    lines = file.readlines()
    map = {}
    idx = 0
    raw = 0
    gene_exp = []
    for line in lines:
        raw += 1
        if raw != 1:
            temp = line.strip('\n').split(',')
            map[idx] = temp[0]
            idx += 1
            gene_exp.append(temp[1:])
    gene_exp = np.array(gene_exp, dtype=float)
    if datasets in nameT:
        gene_exp = gene_exp.T
    print(gene_exp.shape)
    file = open(fp + '/../data/' + datasets + '_label.txt')
    lines = file.readlines()
    label = []
    for line in lines:
        label.append(line.strip('\n'))
    label = np.array(label, dtype=int).reshape(-1)

    return gene_exp, label
