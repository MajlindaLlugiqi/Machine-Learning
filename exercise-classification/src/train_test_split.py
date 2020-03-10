from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

#split the data 20, 80
def trainTestSplit(X, Y):
    #random_state is the seed used by the random number generator
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    return X_train, X_test, y_train, y_test

def kFoldTrainTestSplit(X, Y):
    #random_state is the seed used by the random number generator
    kf = KFold(n_splits=10, shuffle=False, random_state=None)
    X_train, X_test, y_train, y_test = kf.get_n_splits(X,Y)
    return X_train, X_test, y_train, y_test

def stratifiedkFold(X, Y):
    cv = StratifiedKFold(n_splits=4)
    for train_index, test_index in cv.split(X, Y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    return X_train, X_test, y_train, y_test