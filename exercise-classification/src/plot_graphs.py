import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import scipy.stats as ss
#import missingno as msno
import csv

def boxPlot(datasetFeature, datasetTarget, pic_name,):
    data_plt = pd.concat([datasetTarget, datasetFeature], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=datasetFeature, y=datasetTarget, data=data_plt)
    #fig.axis(ymin=0, ymax=1)
    plt.savefig("./graphs/" + pic_name + ".png")
   # plt.axhline(data_scale.mpg.mean(), color='r', linestyle='dashed', linewidth=2)


def plotFeatureDistribution(datasetFeature, pic_name):
    sns.distplot(datasetFeature)
    plt.savefig("./graphs/congression_voting/" + pic_name + ".png")

def plot_correlation_categorical(dataset, Y, target):
    features = []
    correlationMat = []
    dictionary = dict()
    for key, value in dataset.iteritems():
        if (key != target):
            features.append(key)
            confusion_matrix = pd.crosstab(dataset[key], Y).as_matrix()
            correlationMate = cramers_v(confusion_matrix)
            correlationMat.append(correlationMate)
            dictionary[key] = correlationMate

    dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}

    features = []
    correnMat = []
    for key in dictionary:
        features.append(key)
        correnMat.append(dictionary[key])
    y_pos = np.arange(len(features))
    plt.barh(y_pos, correnMat, align='center', alpha=0.5)
    plt.yticks(y_pos, features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.show()


def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def plotBestFeatureCorrolationMatrix(dataset, features):
    # Some features seem highly correlated with Target. Let's plot them separately.
    corr_features = features
    corr_matrix = dataset[corr_features].corr()
    ax = sns.heatmap(corr_matrix, annot=True, square=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Correlation Matrix Heat Map - "best" features')
    plt.show()

def plotPieChart(datasetFeature, directory, pic_name):
    classFrequency = datasetFeature.index.value_counts()
    plt.pie(classFrequency.values, labels=classFrequency.keys(), startangle=90, autopct='%.1f%%')
    plt.title('Frequency of classes')
    plt.tight_layout()
    plt.savefig("./graphs/"+directory+"/" + pic_name + ".png")
    plt.clf()

def plotFeatureHistogram(datasetFeature, directory, pic_name):
    plt.figure(figsize=(15, 8))
    plt.title(pic_name)
    temp = sns.distplot(datasetFeature, bins=30)
    plt.savefig("./graphs/"+directory+"/" + pic_name + ".png")

def plotFeatureImportance(X, Y, directory, pic_name):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(30).plot(kind='barh')
    plt.savefig("./graphs/"+directory+"/" + pic_name + ".png")

def plotDistributionHistogram (dataset, directory, pic_name):
    plt.rcParams.update({'font.size': 20})
    dataset.hist(figsize=(30, 20))
    plt.savefig("./graphs/"+directory+"/" + pic_name + ".png")


def plotCorrelationMatrixHeatMap(dataset, pic_name):
    plt.rcParams.update({'font.size': 35})
    corrmat = dataset.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(50, 50))
    g = sns.heatmap(dataset[top_corr_features].corr())
    plt.savefig("./graphs/image_segmentation/" + pic_name + ".png")

def plotMissingValuesHeatMap(dataset, pic_name):
    #sns.heatmap(dataset.isnull(), cbar=False)
    #plt.figure(figsize=(20, 20))
    #msno.heatmap(dataset,)
    #sns.heatmap(dataset.isnull(), cbar=False)
    #msno.bar(dataset)
   ########## msno.heatmap(dataset)
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
    plt.savefig("./graphs/" + pic_name + ".png")


def plotTimingGroupedBarChart():
    barWidth = 0.25

    # set height of bar
    bars1 = [1.89, 0.03, 0.0021 ,5.8]
    bars2 = [6, 0.38, 0.550095,11.2]
    bars3 = [0.12, 0.005, 0.000897 ,4.1]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='KNN')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='RF')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='NB')

    # Add xticks on the middle of the group bars
    plt.xlabel('Running Time', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Bank ', 'Image ', 'Congressional', 'Amazon'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()


