from sklearn.preprocessing import StandardScaler, MinMaxScaler

# A function that changes the values to a standard scaler
def normalizeStandardScale(train_set, test_set):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(train_set)
   # print(scaler.transform([[2, 2]]))
    X_train = scaler.transform(train_set, None)
    X_test = scaler.transform(test_set, None)
    #return train_set, test_set
    return X_train, X_test

def minMaxScailing(train_set, test_set):
    scaler = MinMaxScaler(copy = True, feature_range = (0, 1))
    X_train = scaler.fit_transform(train_set)
    X_test = scaler.fit_transform(test_set)
    return X_train, X_test