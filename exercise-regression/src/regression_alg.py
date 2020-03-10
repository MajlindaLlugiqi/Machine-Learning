from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor



#play with parameters fit_intercept, normalize and copy_x
def linear_regression(X_train, Y_train, X_predict, fit_intercept=True, normalize = False, copy_X=True):
    regression = linear_model.LinearRegression(fit_intercept, normalize, copy_X)
    #train the model
    regression.fit(X_train, Y_train)
    #print("Coef: ", regression.coef_)
    #print("Intercept: ", regression.intercept_)
    #make prediction
    return regression.predict(X_predict)


#play with parameters n_estimator and criterion
def random_forest_regressor(X_train, Y_train, X_predict):
    #n_estimators are number of trees that are used
    regression = RandomForestRegressor(n_estimators = 100, criterion= "mse")
    regression.fit(X_train, Y_train)
    return regression.predict(X_predict)


#Multi-layer Perceptron regressor.
def mlp_regressor(X_train, Y_train, X_predict):
    regression = MLPRegressor((30,30,30,30),max_iter=2000, alpha=100)
    regression.fit(X_train, Y_train)
    return regression.predict(X_predict)

#KNeighbors Regressor.
def knn_regressor(X_train, Y_train, X_predict):
    regression = KNeighborsRegressor(n_neighbors=10)
    regression.fit(X_train,Y_train)
    return regression.predict(X_predict)