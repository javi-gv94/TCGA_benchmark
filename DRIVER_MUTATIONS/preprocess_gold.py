import pandas

data = pandas.read_csv("input/Mutation.CTAT.3D.Scores.txt", sep='\t',
                               comment="#", header=0)

data['Significant']= data.iloc[:, [56,57,59]].sum(axis=1)

final = data[data['Significant'] >=2]

gold_standard = final[['gene', 'protein_change', 'CODE']]

gold_standard.to_csv('gold_standards.txt', sep='\t', index=False)
