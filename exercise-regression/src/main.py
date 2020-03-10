import sys
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
import numpy as np

from src.read_data import read_bike_sharing_training_data, read_crime_data, read_bike_sharing_testing_data,read_autoMPG_training_data, read_autoMPG_testing_data
from src.evaluation import evaluationForRegression
from src.scaling import minMaxScailing, normalizeStandardScale
from src.data_handler import trainTestSplit, featureOneHotEncoding
from src.plot_graphs import plotMissingValuesHeatMap, plotDistributionHistogram, plotFeatureImportance, plotFeatureDistribution, boxPlot, plotCorrelationMatrixHeatMap
from src.feature_selection import selectKBest, selectRandomForests, selectDecisionTree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print('halo:')

if __name__ == '__main__':
    main()
    dataset = read_bike_sharing_training_data()

    #dataset = read_autoMPG_training_data()
    #dataset = read_autoMPG_training_data()
    #dataset = read_bike_sharing_testing_data()
    X, Y = dataset

    x_train, x_test, y_train, y_test = trainTestSplit(X, Y)
    #x_train, x_test = selectKBest(x_train, y_train, x_test, y_test, 10)



    #x_train, x_test = minMaxScailing(x_train, x_test)

    #x_train, x_test = selectKBest(x_train, y_train, x_test, y_test, 10)
    #x_train, x_test = selectDecisionTree(x_train, y_train, x_test, y_test, 10)
    x_train, x_test = selectRandomForests(x_train, y_train, x_test, y_test, 10)
    evaluationForRegression(x_train, y_train, x_test, y_test)





