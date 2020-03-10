from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import numpy


#split the data 20, 80
def trainTestSplit(X, Y):
    #random_state is the seed used by the random number generator
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    return X_train, X_test, y_train, y_test


#Drops all the rows in a dataset that have at least one missing value
def dropRow(dataset):
    dataset.dropna(inplace=True)
    return dataset

#Drops all the rows in a dataset that reach a threshold of missing values
def dropRowThreshold(dataset, notNullValues):
    threshold = notNullValues * dataset.shape[0]
    dataset = dataset.replace('?', numpy.NAN)
    dataset = dataset.loc[:, dataset.isnull().sum() <= threshold]#The number of missing features that a row needs to reach in order to be deleted
    return dataset

#Adds the mean of a feature to all the missing values that that feature has.
def addMeanforNumericalMissingValue(datasetFeature):
    datasetFeature.fillna(datasetFeature.mean(),inplace=True)


def featureEncoding(train_dataset, columns):
    le = LabelEncoder()
    for column in columns:
        train_dataset[:, column] = le.fit_transform(train_dataset[:, column])
    return train_dataset

def featureOneHotEncoding(dataset):
    print('----->type dataset -------- ', type(dataset))
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto',handle_unknown='ignore')
    #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(dataset)

    print('----->type onehot_encoded  -------- ', type(onehot_encoded))

    df = pd.DataFrame(data=onehot_encoded[1:, 1:],  # values
                 index = onehot_encoded[1:, 0],  # 1st column as index
                 columns = onehot_encoded[0, 1:])

    print('----->type onehot_encoded  -------- ', type(pd))

    return df