from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from math import sqrt
#from src.classification_alg import linear_regression, random_forest_regressor, mlp_regressor, knn_regressor
import timeit


def confusion_matrix_results(test_labels, predicted_labels):
    return confusion_matrix(test_labels, predicted_labels)

def classification_report_results(test_labels, predicted_labels):
    return classification_report(test_labels, predicted_labels)

def classification_accuracy_score(test_labels, predicted_labels):
    return accuracy_score(test_labels, predicted_labels)

def roc_auc_score_results(test_labels, predicted_labels):
    return roc_auc_score(test_labels, predicted_labels)

def classification_roc_auc_score(test_labels, predicted_labels, proba, greater_label):
    y_true = (test_labels == predicted_labels).astype(int)
    y_scores = proba[:,greater_label]
    return roc_auc_score(y_true, y_scores)


