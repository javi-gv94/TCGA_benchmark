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


data = pandas.read_csv( "input/metrics_ref_by_cancer.txt", sep='\t',
                               comment="#", header=0)

cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PANCAN", "PCPG", "PRAD", "READ",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"]

for cancer in cancer_types:
    filtered_data = data.loc[data['Cancer_type'] == cancer]
    gold_standard = filtered_data.iloc[:, 0]
    gold_standard.to_csv(cancer + '.txt', sep=' ', index=False)
    print cancer, len(gold_standard)