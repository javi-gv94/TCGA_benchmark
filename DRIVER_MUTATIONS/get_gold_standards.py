from __future__ import division
import itertools
import pandas



data = pandas.read_csv("input/mut_refs_all_cancers.txt", sep='\t',
                               comment="#", header=0)

cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PANCAN", "PCPG", "PRAD", "READ",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"]

driver_genes = pandas.read_csv("../input/ALL.txt",
                           comment="#", header=None)
driver_genes = driver_genes.iloc[:, 0].values

filtered_data = {}
all_cancer_genes = []
for cancer in cancer_types:
    filtered_data[cancer] = []


for index, row in data.iterrows():

    # check whether the mutation is located in a cancer gene
    if row['Hugo_Symbol'] in driver_genes:

        print index, len(data)

        for type in row['CODE'].split(','):
            filtered_data[type].append([row['Hugo_Symbol'], row['HGVSp_Short']])
            all_cancer_genes.append([row['Hugo_Symbol'], row['HGVSp_Short']])

# remove duplicates
for key, val in filtered_data.iteritems():
    print key
    val.sort()
    filtered_data[key] = list( val for val, _ in itertools.groupby(val))

for cancer in cancer_types:

    df = pandas.DataFrame(filtered_data[cancer], columns=['gene', 'protein_change'])

    df.to_csv(cancer + '.txt', sep='\t', index=False)
    print cancer, len(df)

#get gold standard fo all cancers

all_cancer_genes.sort()
all_cancer_genes =  list( all_cancer_genes for all_cancer_genes, _ in itertools.groupby(all_cancer_genes))

df = pandas.DataFrame(all_cancer_genes, columns=['gene', 'protein_change'])

df.to_csv('ALL.txt', sep='\t', index=False)