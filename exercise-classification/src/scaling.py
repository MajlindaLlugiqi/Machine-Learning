from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# A function that changes the values to a standard scaler
def normalizeStandardScale(train_set, test_set):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(train_set)
   # print(scaler.transform([[2, 2]]))
    X_train = scaler.transform(train_set, None)
    X_test = scaler.transform(test_set, None)
    #return train_set, test_set
    return X_train, X_test

def minMaxScailing(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X

def robustScailing(X):
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    return X