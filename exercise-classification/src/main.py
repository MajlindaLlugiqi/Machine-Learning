import sys
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
from src.classification_alg import getPredictionData
from src.evaluation import confusion_matrix_results, classification_report_results, classification_accuracy_score, classification_roc_auc_score
#from src.read_data import read_image_segmentation, read_congressional_voting, read_bank_marketing
#from src.scaling import minMaxScailing, normalizeStandardScale
from src.data_handler import  featureOHE, overSampling, underSampling,combinedSampling, getFeatureLabelData
from src.scaling import minMaxScailing, robustScailing, normalizeStandardScale
from src.train_test_split import trainTestSplit,  kFoldTrainTestSplit
from src.plot_graphs import plotMissingValuesHeatMap, plotDistributionHistogram, plotFeatureImportance, plotFeatureDistribution, boxPlot, plotCorrelationMatrixHeatMap
from src.feature_selection import selectKBest, selectRandomForests#, selectDecisionTree
import pandas as pd
from src.train_test_split import  train_test_split, kFoldTrainTestSplit,stratifiedkFold
import matplotlib.pyplot as plt
import seaborn as sns

import time


image_segmentation_train = '../data/segmentation.csv'
congressional_voting_train = '../data/CongressionalVotingID.shuf.train.csv'

def main():
    print('halo:')


def evaluate(url, classColumn, noID, samplingType, normalizeType, selectBest, kBest=10):
    print("----------------------------Preprocessing------------------------------------------")
    start_preprocessing_time = time.time()

    X, Y = getFeatureLabelData(url, classColumn, noID)
    if (samplingType == 'over'):
        X_sampled, Y_sampled = overSampling(X, Y)
        x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)
    elif (samplingType == 'under'):
        X_sampled, Y_sampled = underSampling(X, Y)
        x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)
    elif (samplingType == 'combined'):
        X_sampled, Y_sampled = combinedSampling(X, Y)
        x_train, x_test, y_train, y_test = trainTestSplit(X_sampled, Y_sampled)
    else:
        x_train, x_test, y_train, y_test = trainTestSplit(X.values, Y)
    # Pre processing to be added here, not after the preprocessing_time
    if (normalizeType == 'minmax'):
        x_train = minMaxScailing(x_train)
        x_test = minMaxScailing(x_test)
    elif (normalizeType == 'robust'):
        x_train = robustScailing(x_train)
        x_test = robustScailing(x_test)

    if selectBest:
        x_train, x_test = selectKBest(x_train, y_train, x_test, y_test, kBest)
    preprocessing_time = time.time()
    print("The processing time is: ", preprocessing_time - start_preprocessing_time)
    print("----------------------------end Preprocessing------------------------------------------")
    #
    # print("-----------------------NaiveBayes-----------------------------------------")
    # start_prediction_bayes_time = time.time()
    # nb_label_prediction, proba = getPredictionData("NaiveBayes", x_train, x_test, y_train, y_test,kBest)
    # prediction_bayes_time = time.time()
    # print(confusion_matrix_results(y_test, nb_label_prediction))
    # print(classification_report_results(y_test, nb_label_prediction))
    # print("The accuracy is: ", classification_accuracy_score(y_test, nb_label_prediction))
    # # print('The auc_roc score is: ', classification_roc_auc_score(y_test, nb_label_prediction, proba, 0))
    # print("The prediction time for Naive Bayes is:", prediction_bayes_time - start_prediction_bayes_time)
    # print("-----------------------End NaiveBayes-----------------------------------------")

    print("----------------------------kNeighbours------------------------------------------")
    start_prediction_knn_time = time.time()
    kn_label_prediction, proba = getPredictionData("kNeighbours", x_train, x_test, y_train, y_test, 5)
    prediction_knn_time = time.time()
    print(confusion_matrix_results(y_test, kn_label_prediction))
    print(classification_report_results(y_test, kn_label_prediction))
    print('The accuracy is: ', classification_accuracy_score(y_test, kn_label_prediction))
    # print('The auc_roc score is: ', classification_roc_auc_score(y_test, kn_label_prediction, proba, 0))
    print("The prediction time for kNN is:", prediction_knn_time - start_prediction_knn_time)
    print("----------------------------End kNeighbours------------------------------------------")

if __name__ == '__main__':
    main()

    plotPieChart()
   # evaluate(url=image_segmentation_train, classColumn=0, noID=True, samplingType='', normalizeType='robust', selectBest=True,
              #   kBest=1)

   # evaluate(url = congressional_voting_train, encoding='oneHotEncoding', classColumn = 0, noID = False, samplingType = '', scalingType = 'robust')



