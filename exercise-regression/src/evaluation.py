from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from math import sqrt
from src.regression_alg import linear_regression, random_forest_regressor, mlp_regressor, knn_regressor
import timeit


def mean_absolute_error_results(y_actual, y_predicted):
    return mean_absolute_error(y_actual, y_predicted)

def mean_squared_error_results(y_actual, y_predicted):
    return mean_squared_error(y_actual, y_predicted)

def root_mean_squared_error_results(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))

def r2_score_results(y_actual, y_predicted):
    return r2_score(y_actual, y_predicted)

def median_absolute_error_results(y_actual, y_predict):
    return median_absolute_error(y_actual, y_predict)

def evaluationForRegression(x_train, y_train, x_test, y_test):
    print("-----------------------LinearRegression-----------------------------------------")
    lr_start = timeit.default_timer()
    lr_label_prediction = linear_regression(x_train, y_train, x_test, normalize=True)
    lr_stop = timeit.default_timer()
    print("MAE: {}".format(mean_absolute_error_results(y_test, lr_label_prediction)))
    print("MSE: {}".format(mean_squared_error_results(y_test, lr_label_prediction)))
    print("RMS: {}".format(root_mean_squared_error_results(y_test, lr_label_prediction)))
    print("R2S: {}".format(r2_score_results(y_test, lr_label_prediction)))
    print("MDAE: {}".format(median_absolute_error_results(y_test, lr_label_prediction)))
    #print("Accuracy : {}".accuracy_score(y_test, lr_label_prediction))
    print("Execution time: {}".format(lr_stop - lr_start))

    print("----------------------------RandomForestRegressor------------------------------------------")
    rfr_start = timeit.default_timer()
    rfr_label_prediction = random_forest_regressor(x_train, y_train, x_test)
    rfr_stop = timeit.default_timer()
    print("MAE: {}".format(mean_absolute_error_results(y_test, rfr_label_prediction)))
    print("MSE: {}".format(mean_squared_error_results(y_test, rfr_label_prediction)))
    print("RMS: {}".format(root_mean_squared_error_results(y_test, rfr_label_prediction)))
    print("R2S: {}".format(r2_score_results(y_test, rfr_label_prediction)))
    print("MDAE: {}".format(median_absolute_error_results(y_test, rfr_label_prediction)))
    print("Execution time: {}".format(rfr_stop - rfr_start))

    print("----------------------------MLP------------------------------------------")
    rfr_start = timeit.default_timer()
    rfr_label_prediction = mlp_regressor(x_train, y_train, x_test)
    rfr_stop = timeit.default_timer()
    print("MAE: {}".format(mean_absolute_error_results(y_test, rfr_label_prediction)))
    print("MSE: {}".format(mean_squared_error_results(y_test, rfr_label_prediction)))
    print("RMS: {}".format(root_mean_squared_error_results(y_test, rfr_label_prediction)))
    print("R2S: {}".format(r2_score_results(y_test, rfr_label_prediction)))
    print("MDAE: {}".format(median_absolute_error_results(y_test, rfr_label_prediction)))
    print("Execution time: {}".format(rfr_stop - rfr_start))


    print("----------------------------KNN------------------------------------------")
    #rfr_start = timeit.default_timer()
    #rfr_label_prediction = knn_regressor(x_train, y_train, x_test)
    #rfr_stop = timeit.default_timer()
    #print("MAE: {}".format(mean_absolute_error_results(y_test, rfr_label_prediction)))
    #print("MSE: {}".format(mean_squared_error_results(y_test, rfr_label_prediction)))
    #print("RMS: {}".format(root_mean_squared_error_results(y_test, rfr_label_prediction)))
    #print("R2S: {}".format(r2_score_results(y_test, rfr_label_prediction)))
    #print("MDAE: {}".format(median_absolute_error_results(y_test, rfr_label_prediction)))
    #print("Execution time: {}".format(rfr_stop - rfr_start))

