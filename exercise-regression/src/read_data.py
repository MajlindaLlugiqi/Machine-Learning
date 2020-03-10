#import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv
import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

from src.scaling import minMaxScailing
from src.data_handler import trainTestSplit
from sklearn.impute import SimpleImputer
from src.data_handler import dropRowThreshold, dropRow, featureOneHotEncoding
from mlxtend.preprocessing import minmax_scaling
from src.plot_graphs import plotCorrelationMatrixHeatMap, plotBestFeatureCorrolationMatrix, plotFeatureImportance, plotCorrelationMatrixHeatMap, plotMissingValuesHistogram, plotFeatureHistogram, plotFeatureDistribution


def read_bike_sharing_training_data():
    dataset = pd.read_csv('../data/bikeSharing.shuf.train.csv')
    #we drop dteday and id beacuse we can't predict any thing from date or from ID
   # dataset = dataset.drop(['dteday', 'id'], axis=1)
   # dataset = dataset.drop(['dteday'], axis=1)
    X = dataset.drop(['cnt'], axis=1) # axis = 1 -> columns, whereas axis=0 index
    Y = dataset['cnt']
    #plotCorrelationMatrixHeatMap(dataset, 'bike_sharing_matrix_heatmap')
    features = ['hr', 'temp' ,'season', 'cnt']
    plotBestFeatureCorrolationMatrix(dataset, features)
    #dataset.to_csv('read_bike_sharing_scaled.csv')
    return X, Y


def read_bike_sharing_testing_data():
    dataset = pd.read_csv('../data/bikeSharing.shuf.test.csv')
    #we drop dteday and id beacuse we can't predict any thing from date or from ID
    #dataset = dataset.drop(['dteday', 'id'], axis=1)
    dataset = dataset.drop(['dteday'], axis=1)
    #X = dataset.drop(['cnt'], axis=1) # axis = 1 -> columns, whereas axis=0 index
   # Y = dataset['cnt']
    df_scaled = minmax_scaling(dataset, columns=list(dataset.columns))
    df_scaled.to_csv('read_bike_sharing_test__scaled.csv')
    return df_scaled

def read_autoMPG_training_data():
    dataset = pd.read_csv('../data/AutoMPG.shuf.train.csv')
    dataset = dataset.replace('?', numpy.NAN)
   # plotMissingValuesHistogram(dataset, 'auto_missing_values')
   # dataset = dataset.drop(['carName', 'id'], axis=1)
    dataset = dropRow(dataset)
    X = dataset.drop(['mpg'], axis=1) # axis = 1 -> columns, whereas axis=0 index
    Y = dataset['mpg']

    features = ['horsepower','displacement', 'acceleration', 'modelYear', 'mpg']
    plotBestFeatureCorrolationMatrix(dataset, features)

    return X, Y

def read_autoMPG_testing_data():
    dataset = pd.read_csv('../data/AutoMPG.shuf.test.csv')
    dataset = dataset.replace('?', numpy.NAN)
   # plotMissingValuesHistogram(dataset, 'auto_missing_values')
    dataset = dataset.drop(['carName', 'id'], axis=1)
    dataset = dropRow(dataset)
    #X = dataset.drop(['mpg'], axis=1) # axis = 1 -> columns, whereas axis=0 index
   # Y = dataset['mpg']
    #plotFeatureImportance(X, Y, 'feature_importance')
    #features = ['cylinders', 'weight', 'horsepower', 'acceleration', 'mpg']
   # plotBestFeatureCorrolationMatrix(dataset, features)
    #plotCorrelationMatrixHeatMap(dataset,'autoMPG_matrix_heatmap')


    return dataset

def read_crime_data():
    dataset = pd.read_csv('../data/communitiesAndCrime.csv')
    # drop where count of notnull values is larger then 80%:
    print("dataset---->", dataset.shape)
    dataset = dataset.drop(['State','communityname', 'communityCode', 'countyCode', 'murdPerPop', 'rapes',
                           'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults', 'assaultPerPop', 'larcenies',
                           'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'violentPerPop',
                           'nonViolPerPop', 'otherPerCap', 'burglaries', 'burglPerPop','numPolice'], axis=1)
    dataset = dataset.replace('?', numpy.NAN)
    print("dataset--manualy dropped-->", dataset.shape)
    dataset = dropRowThreshold(dataset, 0.8)
    # drop rows with a mising value
   #  dataset = dropRow(dataset)
    X = dataset.drop(['murders'], axis=1)
    Y = dataset['murders']

    dataset = dataset.iloc[:,1:len(dataset)-1]
    for col in dataset.columns:
       # print(col)
        dataset[col] = dataset[col].fillna(dataset[col].mean())

   # features = ['pop', 'pctWhite', 'pctBlack', 'persPoverty', 'murders']
    #plotBestFeatureCorrolationMatrix(dataset, features)


    return X, Y



