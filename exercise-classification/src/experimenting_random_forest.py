import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier

def load_csv_tail(main, tail_list):
    output = []
    for tail in tail_list:
        output.append(pd.read_csv(main + tail + '.csv').to_numpy())
    return output

def get_X(data):
    return data[:, :-1]


def get_y(data):
    return data[:, -1]


# Data set import
# input_file_main = '../data/CongressionalVotingID.shuf.train.'
input_file_main = '../data/BankFull_OHE.csv'
input_file_robust = '../data/BankFull_OHE.shuf.lrn.minMax.csv'
input_file_minmax = '../data/BankFull_OHE.shuf.lrn.robust.csv'
#
# oneHot, underS, overS, combS = load_csv_tail(input_file_main,
#                                              ['1Hot', '1Hot.underSamp', '1Hot.overSamp', '1Hot.combSamp'])

    # main, robust, minmax = load_csv_tail(input_file_main,
    #                                              ['1Hot', '1Hot.underSamp', '1Hot.overSamp', '1Hot.combSamp'])

#
my_random_state = 5

# K-fold Cross Validation
# split = KFold(n_splits=10, shuffle=True, random_state=my_random_state)
my_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=my_random_state)
runtime = []
def main():
    minMax = pd.read_csv(input_file_minmax).to_numpy()  # Add your minmaxed dataset as Numpy array here!
    robust = pd.read_csv(input_file_robust).to_numpy()  # Add your Robust dataset as Numpy array here!
    kSplit = KFold(n_splits=10, shuffle=True, random_state=my_random_state)
    stratKSplit = StratifiedKFold(n_splits=10, shuffle=True, random_state=my_random_state)
    df = pd.read_csv(input_file_main)
    data = df.to_numpy()
    run_dist_test(get_X(data), get_y(data),n_max_estimators=100)

    # run_split_test(data_list=[minMax, robust], split_list=[kSplit, stratKSplit],
    #                 labels=['MinMax & kSplit', 'MinMax & StratSplit', 'Robust & kSplit', 'Robust & StratSplit'],
    #                 title='Split & Scaling Tests',x_axis_text="Nr of trees")
    average = Average(runtime)

    print(round(average, 2))
def Average(lst):
    return sum(lst) / len(lst)
# Parameter testing: Weights & Distances: function
def RF_test_para(X, y, split,n_max_estimators=100, criterion="entropy", max_depth = "None"):
    scores = []
    for my_n_trees in range(1, n_max_estimators, 10):
        split_scores = []

        for train_index, test_index in split.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
           # RandomForestClassifier()
            start_time = time.time()
            model = RandomForestClassifier(n_estimators = n_max_estimators,criterion=criterion)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Scoring
            # split_scores.append(f1_score(y_test, y_pred, average='macro'))
            end_time = time.time()
            runtime.append(end_time - start_time)
            split_scores.append(accuracy_score(y_test, y_pred))

        scores.append([np.mean(split_scores), np.std(split_scores)])

    return np.transpose(scores)


def plot_my_scores(y_values_list, legend_labels, x_axis_text, x_values=range(120,130), title='', y_low_lim=0.85):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle(title, y=0.99)
   #print(y_values_list)
    #print(len(y_values_list))


    axs[0].set_ylim(y_low_lim, 1)
    for i in range(len(y_values_list)):
        for j in [0, 1]:
            axs[j].plot(x_values, y_values_list[i][j], label=legend_labels[i])

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=4)
    axs[1].xaxis.set_label_text(x_axis_text)
    axs[0].yaxis.set_label_text('Accuracy Score - Mean')
    axs[1].yaxis.set_label_text('Accuracy Score - Stdv')
    fig.autofmt_xdate()
   # plt.savefig('../plots/' + title + '.png')
    plt.show()


# Parameter testing: Weights & Distances: plot
def run_dist_test(X, y, n_max_estimators=200):
    scores = [RF_test_para(X, y, my_split, n_max_estimators=n_max_estimators, criterion = 'gini'),
              RF_test_para(X, y, my_split, n_max_estimators=n_max_estimators, criterion = 'entropy'),
              RF_test_para(X, y, my_split, n_max_estimators=n_max_estimators, criterion='entropy',max_depth=5),
              RF_test_para(X, y, my_split, n_max_estimators=n_max_estimators, criterion='entropy', max_depth=50)]

    labels = ['Gini', 'Entropy', 'Entropy & max_depth = 5', 'Entropy & max_depth = 50']
    title = 'Bank Marketing - Random Forest - parameter comparison - criterion'

    plot_my_scores(x_values=range(0, 200, 20), y_values_list=scores, legend_labels=labels,
                   x_axis_text='Number of Trees', title=title)


# Split and Scaling tests
def run_split_test(data_list, split_list, labels, title, x_axis_text='Number of Trees'):
    #print(data_list.count())
    scores = [RF_test_para(get_X(data_list[0]), get_y(data_list[0]), split_list[0]),
              RF_test_para(get_X(data_list[1]), get_y(data_list[1]), split_list[0]),
              RF_test_para(get_X(data_list[0]), get_y(data_list[0]), split_list[1]),
              RF_test_para(get_X(data_list[1]), get_y(data_list[1]), split_list[1])]

    plot_my_scores(y_values_list=scores, legend_labels=labels, x_axis_text=x_axis_text, title=title)

if __name__ == '__main__':
    main()


