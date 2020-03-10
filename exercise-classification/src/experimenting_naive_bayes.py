import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from src.train_test_split import trainTestSplit,  kFoldTrainTestSplit

from sklearn.neighbors import KNeighborsClassifier
from src.evaluation import confusion_matrix_results, classification_report_results, classification_accuracy_score, classification_roc_auc_score

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, average_precision_score
from src.train_test_split import stratifiedkFold

import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
def load_csv_tail(main, tail_list):
    output = []
    for tail in tail_list:
        output.append(pd.read_csv(main + tail + '.csv').to_numpy())
    return output

def get_X(data):
    return data[:, 1:]


def get_y(data):
    return data[:, 0]


# Data set import
# input_file_main = '../data/CongressionalVotingID.shuf.train.'
input_file_main = '../data/Segmentation.csv'
input_file_robust = '../data/minmaxScaling_Segmentation.csv'
input_file_minmax = '../data/robustScaling_Segmentation.csv'
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

    #run_dist_test(get_X(data), get_y(data),n_max_estimators=100)
    X_train, X_test, y_train, y_test = stratifiedkFold(get_X(data), get_y(data)),
    start_time = time.time()
    gnb = GaussianNB()
    # model = RandomForestClassifier(n_estimators = n_max_estimators,criterion=criterion)
    y_pred = gnb.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # Scoring
    # split_scores.append(f1_score(y_test, y_pred, average='macro'))
    end_time = time.time()
    print(""+confusion_matrix_results(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None,
                             zero_division='warn')
    print("f1 :", f1)
    average_precision = average_precision_score(y_test, y_pred)
    print("average_precision :", average_precision)
    print(classification_report_results(y_test, y_pred))
    print("The accuracy is: ", classification_accuracy_score(y_test, y_pred))
    # print('The auc_roc score is: ', classification_roc_auc_score(y_test, nb_label_prediction, proba, 0))
    print("The prediction time for Naive Bayes is:", round(end_time-start_time, 2))
def Average(lst):
    return sum(lst) / len(lst)
# Parameter testing: Weights & Distances: function
def NB_test_para(X, y, split,):
    scores = []
    split_scores = []
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # RandomForestClassifier()
        start_time = time.time()
        gnb = GaussianNB()
        # model = RandomForestClassifier(n_estimators = n_max_estimators,criterion=criterion)
        y_pred = gnb.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        # Scoring
        # split_scores.append(f1_score(y_test, y_pred, average='macro'))
        end_time = time.time()
        runtime.append(end_time - start_time)
        split_scores.append(accuracy_score(y_test, y_pred))

    scores.append([np.mean(split_scores), np.std(split_scores)])

    return np.transpose(scores)



if __name__ == '__main__':
    main()


