import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt
import time
from src.plot_graphs import plotTimingGroupedBarChart

def load_csv_tail(main, tail_list):
    output = []
    for tail in tail_list:
        output.append(pd.read_csv(main + tail + '.csv').to_numpy())
    return output


def get_X(data):
    return data[:, :-1]


def get_y(data):
    return data[:, -1]


def get_best_features(X, y, technique='kBest', k=2500):
    if technique is 'kBest':
        X = SelectKBest(chi2, k=k).fit_transform(X, y)

    if technique is 'tree':
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, y)
        model = SelectFromModel(clf, prefit=True)
        X = model.transform(X)
    return X


def plot_my_scores(x_values, y_values_list, legend_labels, x_axis_text, fixed_y_lim=True, y_lim=(0, 1),
                   title='', ):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(set_title + title, y=0.99)

    if y_lim is fixed_y_lim:
        axs[0].set_ylim(y_lim[0], y_lim[1])

    for i in range(len(y_values_list)):
        for j in [0, 1]:
            axs[j].plot(x_values, y_values_list[i][j], label=legend_labels[i])

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=4)
    axs[1].xaxis.set_label_text(x_axis_text)
    axs[0].yaxis.set_label_text('Accuracy Score - Mean')
    axs[1].yaxis.set_label_text('Accuracy Score - Stdv')
    fig.autofmt_xdate()
    #plt.savefig('../plots/' + "set_title.png")
    plt.show()


# Data set import
# Amazon
input_file_main = '../data/BankFull_OHE.shuf.lrn.'
set_title = 'Bank '
setA, setB = load_csv_tail(input_file_main, ['minMax', 'robust'])

# Voting
# input_file_main = '../data/CongressionalVotingID.shuf.train.'
# set_title = 'Voting '
# setA, setB = load_csv_tail(input_file_main, ['1Hot', '1Hot.underSamp'])

# init
my_random_state = 5
ros = RandomOverSampler(random_state=my_random_state)
my_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=my_random_state)


runtime = []
# Parameter testing:
def RFC_test_para(X, y, split, paras_x_axis, paras, feature_select_tech='none', over_sample=False):
    print('Running RFC_test_para..')
    scores = []
    X = get_best_features(X, y, feature_select_tech)

    for a in paras_x_axis:
        split_scores = []


        for train_index, test_index in split.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if over_sample:
                X_train, y_train = ros.fit_resample(X_train, y_train)

            start_time = time.time()
            model = RandomForestClassifier(n_estimators=a,
                                           max_depth=paras[0],
                                           random_state=my_random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            end_time = time.time()
            runtime.append(end_time - start_time)

            print('Split')
            # Scoring
            # split_scores.append(f1_score(y_test, y_pred, average='macro'))
            split_scores.append(accuracy_score(y_test, y_pred))

        print('Runtime parameter', a, ':', np.mean(runtime))
        scores.append([np.mean(split_scores), np.std(split_scores)])
    output = np.transpose(scores)
    result = np.where(output == np.amax(output))
    my_index = result[1][0]
    print('Best index, mean, stdv: ', my_index, output[:, my_index])

    return output


# Parameter testing: plot
def run_test_para_dist(X, y, paras_x_axis, paras, feature_select_tech='tree'):
    scores = []
    labels = []
    for b in paras:
        scores.append(RFC_test_para(X, y, my_split, paras_x_axis=paras_x_axis, paras=[b], feature_select_tech=feature_select_tech))
        labels.append('max depth: ' + b.__str__())

    title = '- Random Forest - Parameter Comparison'

    plot_my_scores(x_values=paras_x_axis, y_values_list=scores, legend_labels=labels,
                   x_axis_text='Number of Trees', title=title, fixed_y_lim=False, y_lim=(0.9, 1))

def main():
    #run_test_para_dist(get_X(setA), get_y(setA), paras_x_axis=[250], paras=[25])
   # X, y, split, paras_x_axis, paras, feature_select_tech = 'none', over_sample = False
   # RFC_test_para(get_X(setA), get_y(setA), my_split,paras = [25], over_sample = True)
# main
   # average = Average(runtime)
    plotTimingGroupedBarChart()
    #print(round(average, 2))


def Average(lst):
    return sum(lst) / len(lst)

if __name__ == '__main__':
    main()