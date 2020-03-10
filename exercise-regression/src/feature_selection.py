from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def selectKBest(x_train_ds, y_train_ds, x_test_ds, y_test_ds, kBest):
    bestfeatures = SelectKBest(score_func=chi2, k=kBest)
    x_train = bestfeatures.fit_transform(x_train_ds,y_train_ds)
    x_test = bestfeatures.fit_transform(x_test_ds,y_test_ds)
    return x_train, x_test

def selectRandomForests(x_train_ds, y_train_ds, x_test_ds, y_test_ds, max_features):
    x_train = SelectFromModel(RandomForestClassifier(n_estimators = 100), max_features=max_features)
    x_train = x_train.fit_transform(x_train_ds, y_train_ds)
    x_test = SelectFromModel(RandomForestClassifier(n_estimators = 100), max_features=max_features)
    x_test = x_test.fit_transform(x_test_ds, y_test_ds)
    return x_train, x_test

def selectDecisionTree(x_train_ds, y_train_ds, x_test_ds, y_test_ds, max_features):
    x_train = SelectFromModel(ExtraTreesClassifier(n_estimators = 100), max_features=max_features)
    x_train = x_train.fit_transform(x_train_ds, y_train_ds)
    x_test = SelectFromModel(ExtraTreesClassifier(n_estimators = 100), max_features=max_features)
    x_test = x_test.fit_transform(x_test_ds, y_test_ds)
    return x_train, x_test
