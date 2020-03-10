import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import time


def load_csv_tail(main, tail_list):
    output = []
    for tail in tail_list:
        output.append(pd.read_csv(main + tail + '.csv').to_numpy())
    return output

# def get_X(data):
#     return data[:, 1:]
#
#
# def get_y(data):
#     return data[:, 0]
def get_X(data):
    return data[:, :-1]


def get_y(data):
    return data[:, -1]


# Data set import
# input_file_main = '../data/CongressionalVotingID.shuf.train.'
# input_file_main = '../data/Segmentation.csv'
# # input_file_robust = '../data/minmaxScaling_Segmentation.csv'
# # input_file_minmax = '../data/robustScaling_Segmentation.csv'
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
    run_dist_test(get_X(data), get_y(data),n_max_neighbors=50)
    # run_split_test(data_list=[minMax, robust], split_list=[kSplit, stratKSplit],
    #                labels=['MinMax & kSplit', 'MinMax & StratSplit', 'Robust & kSplit', 'Robust & StratSplit'],
    #                title='Split & Scaling Tests',x_axis_text="Nr of neighbors")
    average = Average(runtime)

    print(round(average, 2))


def Average(lst):
    return sum(lst) / len(lst)

# Parameter testing: Weights & Distances: function
def KNN_test_para(X, y, split, n_max_neighbors=10, weights='distance', algorithm='auto', leaf_size=30, p=1,
                  metric='minkowski', metric_params=None):
    scores = []

    for my_n_neighbours in range(1, n_max_neighbors):
        split_scores = []

        for train_index, test_index in split.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = KNeighborsClassifier(n_neighbors=my_n_neighbours, weights=weights, algorithm=algorithm,
                                         leaf_size=leaf_size, p=p, metric=metric, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Scoring
            # split_scores.append(f1_score(y_test, y_pred, average='macro'))
            split_scores.append(accuracy_score(y_test, y_pred))

        scores.append([np.mean(split_scores), np.std(split_scores)])

    return np.transpose(scores)


# Next parameter test
def KNN_test_2(X, y):
    scores = []
    split = StratifiedKFold(n_splits=10, shuffle=True, random_state=my_random_state)

    for my_n_neighbours in range(1, 10):
        split_scores = []

        for train_index, test_index in split.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            start_time = time.time()
            model = KNeighborsClassifier(n_neighbors=my_n_neighbours, weights='distance', p=2)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end_time = time.time()
            runtime.append(end_time - start_time)
            # Scoring
            # split_scores.append(f1_score(y_test, y_pred, average='macro'))
            split_scores.append(accuracy_score(y_test, y_pred))

        scores.append([np.mean(split_scores), np.std(split_scores)])

    return np.transpose(scores)


def plot_my_scores(y_values_list, legend_labels, x_axis_text, x_values=range(1, 10), title='', y_low_lim=0.5):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle(title, y=0.99)

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
def run_dist_test(X, y, n_max_neighbors=50):
    scores = [KNN_test_para(X, y, my_split, n_max_neighbors=n_max_neighbors),
              KNN_test_para(X, y, my_split, n_max_neighbors=n_max_neighbors, weights='distance', p=1),
              KNN_test_para(X, y, my_split, n_max_neighbors=n_max_neighbors, weights='distance'),
              KNN_test_para(X, y, my_split, n_max_neighbors=n_max_neighbors, weights='distance', p=3)]

    labels = ['Uniform', 'Manhattan (p=1)', 'Euclidean (p=2)', 'p=3']
    title = 'Bank marketing - k-NN - parameter comparison - distance measure'

    plot_my_scores(x_values=range(1, n_max_neighbors), y_values_list=scores, legend_labels=labels,
                   x_axis_text='Number of Neigbors', title=title)


# run_dist_test(get_X(oneHot), get_y(oneHot), 50)


# Data tests (used for sample technique testing)
def run_data_test(data_list, labels, title, x_axis_text):
    scores = []

    for data in data_list:
        scores.append(KNN_test_2(get_X(data), get_y(data)))

    plot_my_scores(y_values_list=scores, legend_labels=labels, x_axis_text=x_axis_text, title=title)

my_title = 'Image segmentation - k-NN - sampling comparison'
# run_data_test([oneHot, underS, overS, combS], ['One-Hot only', 'Under Sampled', 'Over Sampled',
#                                               'Combined Sampling'], my_title, x_axis_text='Number of Neighbors')

# Split and Scaling tests
def run_split_test(data_list, split_list, labels, title, x_axis_text='Number of Neighbors'):
    #print(data_list.count())
    scores = [KNN_test_para(get_X(data_list[0]), get_y(data_list[0]), split_list[0]),
              KNN_test_para(get_X(data_list[1]), get_y(data_list[1]), split_list[0]),
              KNN_test_para(get_X(data_list[0]), get_y(data_list[0]), split_list[1]),
              KNN_test_para(get_X(data_list[1]), get_y(data_list[1]), split_list[1])]

    plot_my_scores(y_values_list=scores, legend_labels=labels, x_axis_text=x_axis_text, title=title)

if __name__ == '__main__':
    main()


