from __future__ import division
import numpy as np
import scipy as sp
import scipy.stats
from statsmodels.stats.proportion import proportion_confint
import pandas
import matplotlib.pyplot as plt
import os
import random
import gzip
from sklearn.cluster import KMeans


data = pandas.read_csv("input/score_results.txt", sep='\t',
                               comment="#", header=0)

cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PANCAN", "PCPG", "PRAD", "READ",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"]

filtered_data = {}
for cancer in cancer_types:
    filtered_data[cancer] = []


for index, row in data.iterrows():

    print index, len(data)

    for type in row['CODE'].split(','):
        filtered_data[type].append([row['gene'], row['protein_change']])


for cancer in cancer_types:

    df = pandas.DataFrame(filtered_data[cancer], columns=['gene', 'change'])

    df.to_csv(cancer + '.txt', sep='\t', index=False)
    print cancer, len(df)