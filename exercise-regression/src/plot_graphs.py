import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import missingno as msno
import csv

def boxPlot(datasetFeature, datasetTarget, pic_name):
    data_plt = pd.concat([datasetTarget, datasetFeature], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x='hum', y="cnt", data=data_plt)
    #fig.axis(ymin=0, ymax=1)
    plt.savefig("./graphs/crime/" + pic_name + ".png")
   # plt.axhline(data_scale.mpg.mean(), color='r', linestyle='dashed', linewidth=2)


def plotFeatureDistribution(datasetFeature, pic_name):
    sns.distplot(datasetFeature)
    plt.savefig("./graphs/crime/" + pic_name + ".png")

def plotBestFeatureCorrolationMatrix(dataset, features):
    # Some features seem highly correlated with Target. Let's plot them separately.
    corr_features = features
    corr_matrix = dataset[corr_features].corr()
    ax = sns.heatmap(corr_matrix, annot=True, square=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    #plt.title('Correlation Matrix Heat Map - "best" features')
    plt.show()


def plotFeatureHistogram(datasetFeature, pic_name):
    plt.figure(figsize=(15, 8))
    plt.title(pic_name)
    temp = sns.distplot(datasetFeature, bins=30)
    plt.savefig("./graphs/crime/" + pic_name + ".png")

def plotFeatureImportance(X, Y, pic_name):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(25).plot(kind='barh')
    plt.savefig("./graphs/crime/" + pic_name + ".png")

def plotDistributionHistogram (dataset, pic_name):
    plt.rcParams.update({'font.size': 30})
    dataset.hist(figsize=(30, 20))
    plt.savefig("./graphs/crime/" + pic_name + ".png")


def plotCorrelationMatrixHeatMap(dataset, pic_name):
    plt.rcParams.update({'font.size': 35})
    corrmat = dataset.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(50, 50))
    g = sns.heatmap(dataset[top_corr_features].corr())
    plt.savefig("./graphs/crime/" + pic_name + ".png")

def plotMissingValuesHeatMap(dataset, pic_name):
    #sns.heatmap(dataset.isnull(), cbar=False)
    #plt.figure(figsize=(20, 20))
    #msno.heatmap(dataset,)
    #sns.heatmap(dataset.isnull(), cbar=False)
    #msno.bar(dataset)
    msno.heatmap(dataset)
    #cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
    #sns.heatmap(dataset.isnull(), cmap=cmap, mask=dataset.isnull())
    #msno.matrix(dataset)
    plt.savefig("./graphs/crime/" + pic_name + ".png")


# Creates and saves a histogram for all the missing values of the data set.
# @input dataset - The dataset that you want to create a histogram.
# @input pic_name - The name of the picture generated
def plotMissingValuesHistogram(dataset, pic_name):
    plt.rcParams.update({'font.size': 30})
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum() / dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    f, ax = plt.subplots(figsize=(12, 5))
    plt.figure(figsize=(30, 30))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Percent'])
    missing_data.head()
    plt.xlabel('Features', fontsize=20)
    plt.ylabel('Percent of missing values', fontsize=20)
    plt.title('Percent missing data by feature', fontsize=20)
    plt.savefig("./graphs/crime/" + pic_name + ".png")
