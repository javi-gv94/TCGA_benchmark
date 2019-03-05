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
import math


def compute_metrics(input_dir, gold_standard, cancer_type, all_cancer_genes):

    participants_datasets = {}


    for participant in os.listdir(input_dir + "participants/"):
        # print participant
        if os.path.isfile(input_dir + "participants/" + participant + "/" + cancer_type + ".txt") == False:
            print "#################no data"
            continue

        #read participant predictions and delete insiginificant columns
        predictions = pandas.read_csv("input/participants/" + participant + "/" + cancer_type + ".txt", sep="\t",
                                                      comment="#", header=0)
        if participant == "MutationAssessor":
            predictions = predictions.loc[predictions['score'] >= 3.5]
        elif participant == "SIFT":
            predictions = predictions.loc[predictions['score'] < 0.05]

        predictions.drop(['transcript','score','pvalue','info'], inplace=True, axis=1)
        #drop duplicates
        predictions.drop_duplicates(keep=False, inplace=True)
        # predictions = data[data['qvalue'] <= 0.05]
        # predictions.to_csv(cancer + participant + 'results.txt', sep='\t', index=False)
        #merge both participant and gold standard dataframes in one, and check for overlapping
        df = pandas.merge(predictions, gold_standard, on=['gene', 'protein_change'], how='left', indicator='Overlap')
        df['Overlap'] = np.where(df.Overlap == 'both', True, False)

        final = df[df['Overlap'] == True]

        # final.to_csv(cancer + participant + 'results.txt', sep='\t', index=False)
        # print participant
        # print "predicted", predictions.shape[0]
        # print "overlapping", final.shape[0]
        # print "gold", gold_standard.shape[0]

        #
        all_cancer_genes[participant] = all_cancer_genes[participant].append(predictions[['gene', 'protein_change']], ignore_index=True)
        all_cancer_genes[participant].drop_duplicates(keep=False, inplace=True)

        #
        #number of predicted mutations is the number of rows in participant data
        predicted_mutations = predictions.shape[0]
        #number of overlapping genes with gold standard is the count of overlap=Trues in merged df
        overlapping_genes = final.shape[0]
        #gold standard length
        gold_standard_len = gold_standard.shape[0]

        # TRUE POSITIVE RATE
        TPR = overlapping_genes/gold_standard_len

        #ACCURACY/ PRECISION
        if predicted_mutations == 0:
            acc = 0
        else:
            acc = overlapping_genes / predicted_mutations

        participants_datasets[participant] = [TPR, acc]
        # print participant, TPR, acc
    # print participants_datasets
    return participants_datasets,all_cancer_genes


def pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i, val in enumerate(Xs, 0)], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y
                p_front.append(pair)  # and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]:  # look for lower values of Y
                p_front.append(pair)  # and add them to the pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


def print_chart(participants_datasets, cancer_type):
    tools = []
    x_values = []
    y_values = []
    for tool, metrics in participants_datasets.iteritems():
        tools.append(tool)
        x_values.append(metrics[0])
        y_values.append(metrics[1])

    ax = plt.subplot()
    for i, val in enumerate(tools, 0):
        markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+",
                   "x", "X",
                   "D",
                   "d", "|", "_", ","]
        colors = ['#5b2a49', '#a91310', '#9693b0', '#e7afd7', '#fb7f6a', '#0566e5', '#00bdc8', '#cf4119', '#8b123f',
                  '#b35ccc', '#dbf6a6', '#c0b596', '#516e85', '#1343c3', '#7b88be']

        ax.errorbar(x_values[i], y_values[i], linestyle='None', marker=markers[i],
                    markersize='15', markerfacecolor=colors[i], markeredgecolor=colors[i], capsize=6,
                    ecolor=colors[i], label=tools[i])

    # change plot style
    # set plot title

    plt.title("Cancer Driver Mutations prediction benchmarking - " + cancer_type, fontsize=18, fontweight='bold')

    # set plot title depending on the analysed tool

    ax.set_xlabel("True Positive Rate - % driver mutations correctly predicted", fontsize=12)
    ax.set_ylabel("Precision - % true positives over total predicted", fontsize=12)

    # Shrink current axis's height  on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25,
                     box.width, box.height * 0.75])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), markerscale=0.7,
               fancybox=True, shadow=True, ncol=5, prop={'size': 12})


    # set the axis limits
    x_lims = ax.get_xlim()
    plt.xlim(x_lims)
    y_lims = ax.get_ylim()
    plt.ylim(y_lims)
    if x_lims[0] >= 1000:
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    if y_lims[0] >= 1000:
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, loc: "{:,}".format(int(y))))

    # set parameters for optimization
    better = "top-right"
    max_x = True
    max_y = True

    # get pareto frontier and plot
    p_frontX, p_frontY = pareto_frontier(x_values, y_values, maxX=max_x, maxY=max_y)
    plt.plot(p_frontX, p_frontY, linestyle='--', color='grey', linewidth=1)
    # append edges to pareto frontier
    if better == 'bottom-right':
        left_edge = [[x_lims[0], p_frontX[-1]], [p_frontY[-1], p_frontY[-1]]]
        right_edge = [[p_frontX[0], p_frontX[0]], [p_frontY[0], y_lims[1]]]
        plt.plot(left_edge[0], left_edge[1], right_edge[0], right_edge[1], linestyle='--', color='red',
                 linewidth=1)
    elif better == 'top-right':
        left_edge = [[x_lims[0], p_frontX[-1]], [p_frontY[-1], p_frontY[-1]]]
        right_edge = [[p_frontX[0], p_frontX[0]], [p_frontY[0], y_lims[0]]]
        plt.plot(left_edge[0], left_edge[1], right_edge[0], right_edge[1], linestyle='--', color='red',
                 linewidth=1)

    # add 'better' annotation and quartile numbers to plot
    if better == 'bottom-right':
        plt.annotate('better', xy=(0.98, 0.04), xycoords='axes fraction',
                     xytext=(-30, 30), textcoords='offset points',
                     ha="right", va="bottom",
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.9))
        # my_text1 = plt.text(0.99, 0.15, '1',
        #                     verticalalignment='bottom', horizontalalignment='right',
        #                     transform=ax.transAxes, fontsize=25)
        # my_text2 = plt.text(0.01, 0.15, '2',
        #                     verticalalignment='bottom', horizontalalignment='left',
        #                     transform=ax.transAxes, fontsize=25)
        # my_text3 = plt.text(0.99, 0.85, '3',
        #                     verticalalignment='top', horizontalalignment='right',
        #                     transform=ax.transAxes, fontsize=25)
        # my_text4 = plt.text(0.01, 0.85, '4',
        #                     verticalalignment='top', horizontalalignment='left',
        #                     transform=ax.transAxes, fontsize=25)
        # my_text1.set_alpha(.2)
        # my_text2.set_alpha(.2)
        # my_text3.set_alpha(.2)
        # my_text4.set_alpha(.2)
    elif better == 'top-right':
        plt.annotate('better', xy=(0.98, 0.95), xycoords='axes fraction',
                     xytext=(-30, -30), textcoords='offset points',
                     ha="right", va="top",
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.9))
        # my_text1 = plt.text(0.99, 0.85, '1',
        #                     verticalalignment='top', horizontalalignment='right',
        #                     transform=ax.transAxes, fontsize=25)
        # my_text2 = plt.text(0.01, 0.85, '2',
        #                     verticalalignment='top', horizontalalignment='left',
        #                     transform=ax.transAxes, fontsize=25)
        # my_text3 = plt.text(0.99, 0.01, '3',
        #                     verticalalignment='bottom', horizontalalignment='right',
        #                     transform=ax.transAxes, fontsize=25)
        # my_text4 = plt.text(0.01, 0.01, '4',
        #                     verticalalignment='bottom', horizontalalignment='left',
        #                     transform=ax.transAxes, fontsize=25)

        # my_text1.set_alpha(.2)
        # my_text2.set_alpha(.2)
        # my_text3.set_alpha(.2)
        # my_text4.set_alpha(.2)

    # plot quartiles

    # tools_quartiles_squares = plot_square_quartiles(x_values, y_values, tools, better)
    # tools_quartiles_diagonal = plot_diagonal_quartiles(x_values, y_values, tools, better)
    #
    # tools_clusters = cluster_tools(zip(x_values, y_values), tools, better)


    # plt.show()
    outname = "output/" + cancer_type + "_output.png"
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(outname, dpi=100)

    plt.close("all")

    # return tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters


##############################################################################################################
##############################################################################################################
##############################################################################################################

cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PANCAN", "PCPG", "PRAD", "READ",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"]

#remove PANCAN
cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"]

# cancer_types = ["CHOL"]

input_dir = "input/"



## create dict that will store info about all combined cancer types
all_cancer_genes = {}
for participant in os.listdir(input_dir + "participants/"):
    all_cancer_genes[participant] = pandas.DataFrame(columns=['gene','protein_change'])

# this dictionary will store all the information required for the quartiles table
quartiles_table = {}

for cancer in cancer_types:

    gold_standard = pandas.read_csv("input/METRICS_REFS/"+ cancer + ".txt", sep="\t",
                           comment="#", header=0)


    participants_datasets, all_cancer_genes = compute_metrics(input_dir, gold_standard, cancer,all_cancer_genes)
    print_chart(participants_datasets, cancer)
    # tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters = print_chart(participants_datasets, cancer)
    # quartiles_table[cancer] = [tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters]


# plot chart for results across all cancer types

gold_standard = pandas.read_csv("input/METRICS_REFS/ALL.txt", sep="\t",
                                comment="#", header=0)

participants_datasets = {}
for participant, predicted_mutations in all_cancer_genes.iteritems():

    # print participant, predicted_mutations.shape[0]
    predicted_mutations.drop_duplicates(subset=['gene', 'protein_change'])
    predicted_mutations.to_csv(participant + 'results.txt', sep='\t', index=False)

    # merge both participant and gold standard dataframes in one, and check for overlapping
    df = pandas.merge(predicted_mutations, gold_standard, on=['gene', 'protein_change'], how='left', indicator='Overlap')
    df['Overlap'] = np.where(df.Overlap == 'both', True, False)

    final = df[df['Overlap'] == True]
    # predicted_mutations.to_csv(participant + '_ALLresults.txt', sep='\t', index=False)

    # number of predicted mutations is the number of rows in participant data
    predicted_mutations = predicted_mutations.shape[0]
    # number of overlapping genes with gold standard is the count of overlap=Trues in merged df
    overlapping_genes = final.shape[0]
    # gold standard length
    gold_standard_len = gold_standard.shape[0]

    # TRUE POSITIVE RATE
    TPR = overlapping_genes / gold_standard_len

    # ACCURACY/ PRECISION
    if predicted_mutations == 0:
        acc = 0
    else:
        acc = overlapping_genes / predicted_mutations

    participants_datasets[participant] = [TPR, acc]
    print participant,TPR,acc
    print "predicted", predicted_mutations
    print "overlapping", final.shape[0]
    print "gold", gold_standard.shape[0]

print_chart(participants_datasets, "ALL")
# quartiles_table["ALL"] = [tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters]



