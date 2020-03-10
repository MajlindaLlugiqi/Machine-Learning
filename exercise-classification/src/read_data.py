#import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv
import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from src.train_test_split import  train_test_split, kFoldTrainTestSplit,stratifiedkFold
#from src.scaling import minMaxScailing
from sklearn.impute import SimpleImputer

from src.data_handler import dropRowThreshold, dropRow, featureOHE, combinedSampling, addNominalMissingValueMode
#from mlxtend.preprocessing import minmax_scaling
from src.plot_graphs import plotCorrelationMatrixHeatMap,boxPlot, plot_correlation_categorical,plotPieChart,plotDistributionHistogram, plotBestFeatureCorrolationMatrix, plotFeatureImportance, plotCorrelationMatrixHeatMap, plotMissingValuesHistogram, plotFeatureHistogram, plotFeatureDistribution

def read_bank_marketing():
    dataset = pd.read_csv('../data/bankfull.csv')
    dataset = dataset.replace('unknown', numpy.NAN)
    dataset = dropRowThreshold(dataset, 80)
    dataset['contact'] = addNominalMissingValueMode(dataset['contact'])
    dataset['education'] = addNominalMissingValueMode(dataset['education'])
    dataset['job'] = addNominalMissingValueMode(dataset['job'])

    X = dataset.drop(['y'], axis=1) # axis = 1 -> columns, whereas axis=0 index
    #X = featureOHE(X)
    Y = dataset['y']
    return X, Y

def read_image_segmentation():
    dataset = pd.read_csv('../data/segmentation.csv')
    X = dataset.drop(['CLASS'], axis=1) # axis = 1 -> columns, whereas axis=0 index
    Y = dataset['CLASS']
    #minMaxScailing(X)
    #robustScailing(X)
    return X, Y


def read_amazon_commerce():
    dataset = pd.read_csv('../data/segmentation.csv')
    X = dataset.drop(['CLASS'], axis=1) # axis = 1 -> columns, whereas axis=0 index
    Y = dataset['CLASS']
    #minMaxScailing(X)
    #robustScailing(X)
    return X, Y

def read_congressional_voting():
    dataset = pd.read_csv('../data/CongressionalVotingID.shuf.train.csv')

   # X = featureOneHotEncoding(X)
    X = dataset.drop(['class'], axis=1)  # axis = 1 -> columns, whereas axis=0 index
    Y = dataset['class']
    return X, Y

