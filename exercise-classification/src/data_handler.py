
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
import numpy
#from sklearn.preprocessing import Imputer

#Drops all the rows in a dataset that have at least one missing value
def dropRow(dataset):
    dataset.dropna(inplace=True)
    return dataset

# def dropRowThreshold(dataset, threshold):
#     dataset.dropna(thresh=threshold,inplace=True)
#     return dataset

# #Drops all the rows in a dataset that reach a threshold of missing values
def dropRowThreshold(dataset, notNullValues):
    threshold = notNullValues * dataset.shape[0]/100
    dataset = dataset.loc[:, dataset.isnull().sum() <= threshold]#The number of missing features that a row needs to reach in order to be deleted
    return dataset


#Adds the mean of a feature to all the missing values that that feature has.
def addMeanforNumericalMissingValue(datasetFeature):
    datasetFeature.fillna(datasetFeature.mean(),inplace=True)

def addNominalMissingValueMode(datasetFeature):
    datasetFeature.fillna(datasetFeature.mode()[0], inplace=True)
    return datasetFeature

def overSampling(X, Y):
    #smote = SMOTE(random_state=42)
    s = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = s.fit_resample(X, Y)
    return X_resampled, y_resampled

def underSampling(X, Y):
    nm1 = NearMiss(version=1)
    X_resampled, y_resampled = nm1.fit_resample(X, Y)
    return X_resampled, y_resampled

def combinedSampling(X, Y):
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, Y)
    return X_resampled, y_resampled

def getFeatureLabelData(train_url,classColumn, noIDColumn):
    dataset = pd.read_csv(train_url)
    if noIDColumn:
        X = dataset.drop(dataset.columns[[classColumn]], axis=1)
    else:
        X = dataset.drop(dataset.columns[[0, classColumn]], axis=1)

    Y = dataset.iloc[:, classColumn].values
    return X,Y


def featureEncoding(train_dataset, columns):
    le = LabelEncoder()
    for column in columns:
        train_dataset[:, column] = le.fit_transform(train_dataset[:, column])
    return train_dataset


def featureOHE(X):
    X = pd.get_dummies(X)
    print(X)
    X.to_csv(r'..\\data\\oneHotEncoding.csv', index=None, header=True)
    return X