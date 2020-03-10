from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


#kNN diff neighbors and diff weight
def kNeighbours(n_neighbors, weight, feature_train, label_train, feature_test):
    classifier = KNeighborsClassifier(n_neighbors, weight)
    classifier.fit(feature_train, label_train)
    label_prediction = classifier.predict(feature_test)
    proba = classifier.predict_proba(feature_test)
    return label_prediction, proba

#Naive bayes
def naiveBayes(feature_train, label_train, feature_test):
    classifier = GaussianNB()
    #gnb.fit(X_train, y_train).predict(X_test)
    classifier.fit(feature_train, label_train)
    predicted = classifier.predict(feature_test)
    proba = classifier.predict_proba(feature_test)
    return predicted, proba



def getPredictionData(type, X_train, X_test, Y_train, Y_test, N_NEIGHBORS):
    if (type == "NaiveBayes"):
        label_prediction, proba = naiveBayes(X_train,Y_train, X_test)
    elif (type == "kNeighbours"):
        label_prediction, proba = kNeighbours(N_NEIGHBORS,'uniform', X_train, Y_train, X_test)

    return label_prediction, proba