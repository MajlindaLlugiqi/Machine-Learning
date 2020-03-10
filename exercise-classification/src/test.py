import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import time
#
my_random_state = 5

# Data set import
input_file = '../data/Segmentation.csv'
df = pd.read_csv(input_file)
data = df.to_numpy()

my_X = data[:, :-1]
my_y = data[:, -1]

# K-fold Cross Validation
#my_split = KFold(n_splits=10, shuffle=True, random_state=my_random_state)
#my_split = train_test_split(my_X, my_y,train_size=0.2, random_state=my_random_state)
my_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=my_random_state)
runtime = []
def main():
    # Parameter testing: Weights & Distances: plot
    my_n_max_neighbors = 50

    my_scores = [KNN_para_test(my_X, my_y, my_split, n_max_neighbors=my_n_max_neighbors),
                 KNN_para_test(my_X, my_y, my_split, n_max_neighbors=my_n_max_neighbors, weights='distance', p=1),
                 KNN_para_test(my_X, my_y, my_split, n_max_neighbors=my_n_max_neighbors, weights='distance'),
                 KNN_para_test(my_X, my_y, my_split, n_max_neighbors=my_n_max_neighbors, weights='distance', p=3)]

    my_labels = ['Uniform', 'Manhattan (p=1)', 'Euclidean (p=2)', 'p=3']
    my_title = 'Image Segmentation - k-NN - parameter comparison - distance measure'
    average = Average(runtime)
    print(round(average, 2))
    plot_my_scores(x_values=range(1, my_n_max_neighbors), y_values_list=my_scores, legend_labels=my_labels,
                   x_axis_text='Number of Neigbors', title=my_title)
def Average(lst):
    return sum(lst) / len(lst)

def KNN_para_test(X, y, split, n_max_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                  metric='minkowski', metric_params=None):
    scores = []

    for my_n_neighbours in range(1, n_max_neighbors):
        split_scores = []


        for train_index, test_index in split.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            start_time = time.time()
            model = KNeighborsClassifier(n_neighbors=my_n_neighbours, weights=weights, algorithm=algorithm,
                                         leaf_size=leaf_size, p=p, metric=metric)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end_time = time.time()
            #print("===>time ", end_time - start_time)

            # Scoring
            # split_scores.append(f1_score(y_test, y_pred, average='macro'))
            split_scores.append(accuracy_score(y_test, y_pred))
            runtime.append(end_time - start_time)

        scores.append([np.mean(split_scores), np.std(split_scores)])

    return np.transpose(scores)

# Next parameter test
def KNN_para_test_2():
    scores = []

    return np.transpose(scores)

def plot_my_scores(x_values, y_values_list, legend_labels, x_axis_text, title='', y_low_lim=0.8):
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
    #plt.savefig('../plots/'+'is-raw-kfo.png')
    plt.show()




if __name__ == '__main__':
    main()