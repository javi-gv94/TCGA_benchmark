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



def compute_metrics(input_dir, gold_standard, cancer_type,all_cancer_genes):

    participants_datasets = {}


    for participant in os.listdir(input_dir + "participants/"):

        data = pandas.read_csv(input_dir + "participants/" + participant + "/" + cancer_type + ".txt", sep='\t',
                               comment="#", header=0)

        filtered_data = data.loc[data['qvalue'] <= 0.05]

        predicted_genes = filtered_data.iloc[:, 0].values

        # predicted_genes = data.iloc[:, 0].values

        all_cancer_genes[participant] = list(set().union(predicted_genes, all_cancer_genes[participant]))

        # TRUE POSITIVES
        overlapping_genes = set(predicted_genes).intersection(gold_standard)

        #ACCURACY/ PRECISION
        if len(predicted_genes) == 0:
            acc = 0
        else:
            acc = len(overlapping_genes) / len(predicted_genes)

        participants_datasets[participant] = [len(overlapping_genes), acc]

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

# funtion that gets quartiles for x and y values
def plot_square_quartiles(x_values, means, tools, better, percentile=50):
    x_percentile, y_percentile = (np.nanpercentile(x_values, percentile), np.nanpercentile(means, percentile))
    # plt.axvline(x=x_percentile, linestyle='-', color='black', linewidth=0.1)
    # plt.axhline(y=y_percentile, linestyle='-', color='black', linewidth=0.1)

    # create a dictionary with tools and their corresponding quartile
    tools_quartiles = {}
    if better == "bottom-right":
        for i, val in enumerate(tools, 0):
            if x_values[i] >= x_percentile and means[i] <= y_percentile:
                tools_quartiles[tools[i]] = 1
            elif x_values[i] >= x_percentile and means[i] > y_percentile:
                tools_quartiles[tools[i]] = 3
            elif x_values[i] < x_percentile and means[i] > y_percentile:
                tools_quartiles[tools[i]] = 4
            elif x_values[i] < x_percentile and means[i] <= y_percentile:
                tools_quartiles[tools[i]] = 2
    elif better == "top-right":
        for i, val in enumerate(tools, 0):
            if x_values[i] >= x_percentile and means[i] < y_percentile:
                tools_quartiles[tools[i]] = 3
            elif x_values[i] >= x_percentile and means[i] >= y_percentile:
                tools_quartiles[tools[i]] = 1
            elif x_values[i] < x_percentile and means[i] >= y_percentile:
                tools_quartiles[tools[i]] = 2
            elif x_values[i] < x_percentile and means[i] < y_percentile:
                tools_quartiles[tools[i]] = 4
    return (tools_quartiles)


# function to normalize the x and y axis to 0-1 range
def normalize_data(x_values, means):
    maxX = max(x_values)
    minX = min(x_values)
    maxY = max(means)
    minY = min(means)
    # maxX = ax.get_xlim()[1]
    # minX = ax.get_xlim()[0]
    # maxY = ax.get_ylim()[1]
    # minY = ax.get_ylim()[0]
    # x_norm = [(x - minX) / (maxX - minX) for x in x_values]
    # means_norm = [(y - minY) / (maxY - minY) for y in means]
    x_norm = [x / maxX for x in x_values]
    means_norm = [y / maxY for y in means]
    return x_norm, means_norm


# funtion that plots a diagonal line separating the values by the given quartile
def draw_diagonal_line(scores_and_values, quartile, better, max_x, max_y):
    for i, val in enumerate(scores_and_values, 0):
        # find out which are the two points that contain the percentile value
        if scores_and_values[i][0] <= quartile:
            target = [(scores_and_values[i - 1][1], scores_and_values[i - 1][2]),
                      (scores_and_values[i][1], scores_and_values[i][2])]
            break
    # get the the mid point between the two, where the quartile line will pass
    half_point = (target[0][0] + target[1][0]) / 2, (target[0][1] + target[1][1]) / 2
    # plt.plot(half_point[0], half_point[1], '*')
    # draw the line depending on which is the optimal corner
    if better == "bottom-right":
        x_coords = (half_point[0] - max_x, half_point[0] + max_x)
        y_coords = (half_point[1] - max_y, half_point[1] + max_y)
    elif better == "top-right":
        x_coords = (half_point[0] + max_x, half_point[0] - max_x)
        y_coords = (half_point[1] - max_y, half_point[1] + max_y)

    plt.plot(x_coords, y_coords, linestyle='--', linewidth=1.5)


# funtion that splits the analysed tools into four quartiles, according to the asigned score
def get_quartile_points(scores_and_values, first_quartile, second_quartile, third_quartile):
    tools_quartiles = {}
    for i, val in enumerate(scores_and_values, 0):
        if scores_and_values[i][0] > third_quartile:
            tools_quartiles[scores_and_values[i][3]] = 1
        elif second_quartile < scores_and_values[i][0] <= third_quartile:
            tools_quartiles[scores_and_values[i][3]] = 2
        elif first_quartile < scores_and_values[i][0] <= second_quartile:
            tools_quartiles[scores_and_values[i][3]] = 3
        elif scores_and_values[i][0] <= first_quartile:
            tools_quartiles[scores_and_values[i][3]] = 4
    return (tools_quartiles)


# funtion that separate the points through diagonal quartiles based on the distance to the 'best corner'
def plot_diagonal_quartiles(x_values, means, tools, better):
    # get distance to lowest score corner

    # normalize data to 0-1 range
    x_norm, means_norm = normalize_data(x_values, means)
    max_x = max(x_values)
    max_y = max(means)
    # compute the scores for each of the tool. based on their distance to the x and y axis
    scores = []
    for i, val in enumerate(x_norm, 0):
        if better == "bottom-right":
            scores.append(x_norm[i] + (1 - means_norm[i]))
        elif better == "top-right":
            scores.append(x_norm[i] + means_norm[i])

    # add plot annotation boxes with info about scores and tool names
    for counter, scr in enumerate(scores):
        plt.annotate(
            tools[counter] + "\n" +
            # str(round(x_norm[counter], 6)) + " * " + str(round(1 - means_norm[counter], 6)) + " = " + str(round(scr, 8)),
            "score = " + str(round(scr, 3)),
            xy=(x_values[counter], means[counter]), xytext=(0, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.15),
            size=7,
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # region sort the list in descending order
    scores_and_values = sorted([[scores[i], x_values[i], means[i], tools[i]] for i, val in enumerate(scores, 0)],
                               reverse=True)
    scores = sorted(scores, reverse=True)
    # print (scores_and_values)
    # print (scores)
    # endregion
    first_quartile, second_quartile, third_quartile = (
        np.nanpercentile(scores, 25), np.nanpercentile(scores, 50), np.nanpercentile(scores, 75))
    # print (first_quartile, second_quartile, third_quartile)
    draw_diagonal_line(scores_and_values, first_quartile, better, max_x, max_y)
    draw_diagonal_line(scores_and_values, second_quartile, better, max_x, max_y)
    draw_diagonal_line(scores_and_values, third_quartile, better, max_x, max_y)

    # split in quartiles
    tools_quartiles = get_quartile_points(scores_and_values, first_quartile, second_quartile, third_quartile)
    return (tools_quartiles)


def cluster_tools(my_array, tools, better):
    X = np.array(my_array)
    kmeans = KMeans(n_clusters=4, n_init=50, random_state=0).fit(X)
    # print (method, organism)
    cluster_no = kmeans.labels_

    centroids = kmeans.cluster_centers_

    # normalize data to 0-1 range
    x_values = []
    y_values = []
    for centroid in centroids:
        x_values.append(centroid[0])
        y_values.append(centroid[1])
    x_norm, y_norm = normalize_data(x_values, y_values)
    # plt.plot(centroids[0][0], centroids[0][1], '*')
    # get distance from centroids to better corner
    distances = []
    if better == "top-right":
        best_point = [1, 1]
        for x, y in zip(x_norm, y_norm):
            distances.append(x + y)
    elif better == "bottom-right":
        best_point = [1, 0]
        for x, y in zip(x_norm, y_norm):
            distances.append(x + (1 - y))
    # for i, centroid in enumerate(centroids):
    #     plt.plot(centroid[0], centroid[1], '*', markersize=20)
        # plt.text(centroid[0], centroid[1], distances[i], color="green", fontsize=18)

    # assing ranking to distances array
    output = [0] * len(distances)
    for i, x in enumerate(sorted(range(len(distances)), key=lambda y: distances[y], reverse=True)):
        output[x] = i

    # reorder the clusters according to distance
    for i, val in enumerate(cluster_no):
        for y, num in enumerate(output):
            if val == y:
                cluster_no[i] = num

    tools_clusters = {}
    for (x, y), num, name in zip(X, cluster_no, tools):
        tools_clusters[name] = num + 1
        # plt.text(x, y, num + 1, color="red", fontsize=18)


    return tools_clusters


# function that prints a table with the list of tools and the corresponding quartiles
def print_quartiles_table(tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters):
    row_names = tools_quartiles_squares.keys()
    quartiles_1 = tools_quartiles_squares.values()
    quartiles_2 = []
    clusters = []
    for i, val in enumerate(row_names, 0):
        quartiles_2.append(tools_quartiles_diagonal[row_names[i]])
        clusters.append(tools_clusters[row_names[i]])
    colnames = ["TOOL", "Quartile_diag"] #, "Quartile_diag", "Cluster"]
    celltxt = zip(row_names, quartiles_2) #, quartiles_2, clusters)
    df = pandas.DataFrame(celltxt)
    vals = df.values

    # set cell colors depending on the quartile
    colors = df.applymap(lambda x: '#66cdaa' if x == 1 else '#7fffd4' if x == 2 else '#ffa07a' if x == 3
    else '#fa8072' if x == 4 else '#ffffff')
    # grey color scale
    colors = df.applymap(lambda x: '#919191' if x == 1 else '#B0B0B0' if x == 2 else '#CFCFCF' if x == 3
    else '#EDEDED' if x == 4 else '#ffffff')
    # green color scale
    colors = df.applymap(lambda x: '#238b45' if x == 1 else '#74c476' if x == 2 else '#bae4b3' if x == 3
    else '#edf8e9' if x == 4 else '#ffffff')
    # red color scale
    # colors = df.applymap(lambda x: '#fee5d9' if x == 1 else '#fcae91' if x == 2 else '#fb6a4a' if x == 3
    # else '#cb181d' if x == 4 else '#ffffff')

    colors = colors.values

    the_table = plt.table(cellText=vals,
                          colLabels=colnames,
                          cellLoc='center',
                          loc='right',
                          bbox=[1.1, 0.15, 0.5, 0.8],
                          colWidths=[1.2, 0.5],
                          cellColours=colors,
                          colColours=['#ffffff'] * 2)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    plt.subplots_adjust(right=0.65, bottom=0.2)


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

    plt.title("Cancer Driver Genes prediction benchmarking - " + cancer_type, fontsize=18, fontweight='bold')

    # set plot title depending on the analysed tool

    ax.set_xlabel("True Positives - Num driver genes correctly predicted", fontsize=12)
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
        #
        # my_text1.set_alpha(.2)
        # my_text2.set_alpha(.2)
        # my_text3.set_alpha(.2)
        # my_text4.set_alpha(.2)

    # plot quartiles

    tools_quartiles_squares = plot_square_quartiles(x_values, y_values, tools, better)
    tools_quartiles_diagonal = plot_diagonal_quartiles(x_values, y_values, tools, better)

    tools_clusters = cluster_tools(zip(x_values, y_values), tools, better)

    print_quartiles_table(tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters)

    # plt.show()
    outname = "output/" + cancer_type + "_output.png"
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(outname, dpi=100)

    plt.close("all")

    return tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters


# function that prints a table with the list of tools and the corresponding quartiles
def print_full_table(quartiles_table):
    # (tools_quartiles_squares, tools_quartiles_diagonal, method):
    colnames = ["TOOL / QUARTILES -->"]
    for name in quartiles_table.keys():
        colnames.append("SQR")
        colnames.append("DIAG")
        colnames.append("CLUST  ")
    colnames.extend(["# SQR", "# DIAG", "# CLUST"])
    row_names = quartiles_table[next(iter(quartiles_table))][0].keys()
    quartiles_list = []

    for name in sorted(quartiles_table.iterkeys()):
        quartiles_sqr = []
        quartiles_diag = []
        clusters = []
        for i in row_names:
            # print (name)
            # print (i, quartiles_table[name][0][i])
            # quartiles_sqr.append(quartiles_table[name][0][i])
            quartiles_diag.append(quartiles_table[name][1][i])
            # clusters.append(quartiles_table[name][2][i])
        # quartiles_list.append(quartiles_sqr)
        quartiles_list.append(quartiles_diag)
        # quartiles_list.append(clusters)
    print (quartiles_list)
    text = []
    for tool in row_names:
        text.append([tool])

    for num, name in enumerate(row_names):
        for i in range(len(quartiles_table.keys())):# * 3):
            text[num].append(quartiles_list[i][num])
    print (text)

    # get total score for square and diagonal quartiles
    sqr_quartiles_sums = {}
    diag_quartiles_sums = {}
    cluster_sums = {}
    for num, val in enumerate(text):
        # total_sqr = sum(text[num][i] for i in range(1, len(text[num]), 1))
        total_diag = sum(text[num][i] for i in range(1, len(text[num]), 1))
        # total_clust = sum(text[num][i] for i in range(3, len(text[num]), 3))
        # sqr_quartiles_sums[text[num][0]] = total_sqr
        diag_quartiles_sums[text[num][0]] = total_diag
        # cluster_sums[text[num][0]] = total_clust
    # sort tools by that score to rank them
    # sorted_sqr_quartiles_sums = sorted(sqr_quartiles_sums.items(), key=lambda x: x[1])
    sorted_diag_quartiles_sums = sorted(diag_quartiles_sums.items(), key=lambda x: x[1])
    # sorted_clust_sums = sorted(cluster_sums.items(), key=lambda x: x[1])

    # append to the final table
    # for i, val in enumerate(sorted_sqr_quartiles_sums):
    #     for j, lst in enumerate(text):
    #         if val[0] == text[j][0]:
    #             text[j].append("# " + str(i + 1))
    for i, val in enumerate(sorted_diag_quartiles_sums):
        for j, lst in enumerate(text):
            if val[0] == text[j][0]:
                text[j].append("# " + str(i + 1))
    # for i, val in enumerate(sorted_clust_sums):
    #     for j, lst in enumerate(text):
    #         if val[0] == text[j][0]:
    #             text[j].append("# " + str(i + 1))

    print (text)

    df = pandas.DataFrame(text)
    df1 = df.iloc[:, [0, 32]].copy()
    vals = df1.values
    print(vals)

    ##

    ##
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    fig.tight_layout()

    the_table = ax.table(cellText=vals,
                         colLabels=["TOOL", "#Ranking"],
                         cellLoc='center',
                         loc='center',
                         # bbox=[1.1, 0.15, 0.5, 0.8])
                         colWidths=[0.16, 0.06],
                         )
    fig.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1.5)
    plt.subplots_adjust(right=0.95, left=0.04, top=0.9, bottom=0.1)


##############################################################################################################
##############################################################################################################
##############################################################################################################

cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PANCAN", "PCPG", "PRAD", "READ",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"]

cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCS", "UVM"]

cancer_types = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KICH", "KIRC",
                "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD",
                "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCS", "UVM"]

input_dir = "input/"

# data = pandas.read_csv(input_dir + "metrics_ref.txt", comment="#", header=None)
#
# gold_standard = data.iloc[:, 0].values

## create dict that will store info about all combined cancer types
all_cancer_genes = {}
for participant in os.listdir(input_dir + "participants/"):
    all_cancer_genes[participant] = []

# this dictionary will store all the information required for the quartiles table
quartiles_table = {}

for cancer in cancer_types:

    data = pandas.read_csv("input/"+ cancer + ".txt",
                           comment="#", header=None)
    gold_standard = data.iloc[:, 0].values

    participants_datasets, all_cancer_genes = compute_metrics(input_dir, gold_standard, cancer,all_cancer_genes)
    tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters = print_chart(participants_datasets, cancer)
    quartiles_table[cancer] = [tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters]


# plot chart for results across all cancer types

data = pandas.read_csv("input/ALL.txt",
                           comment="#", header=None)
gold_standard = data.iloc[:, 0].values

participants_datasets = {}
for participant, predicted_genes in all_cancer_genes.iteritems():

    # TRUE POSITIVES
    overlapping_genes = set(predicted_genes).intersection(gold_standard)

    # ACCURACY/ PRECISION
    if len(predicted_genes) == 0:
        acc = 0
    else:
        acc = len(overlapping_genes) / len(predicted_genes)

    participants_datasets[participant] = [len(overlapping_genes), acc]


tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters = print_chart(participants_datasets, "ALL")
quartiles_table["ALL"] = [tools_quartiles_squares, tools_quartiles_diagonal, tools_clusters]


#print summary table across all cancer types
print_full_table(quartiles_table)
# plt.show()
out_table = "output/table.png"
fig = plt.gcf()
fig.set_size_inches(20, 11.1)
fig.savefig(out_table, dpi=100)

plt.close("all")