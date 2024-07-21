import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NestedCrossValidation import NestedCrossValidation
import joblib

#
# Settings
#
APPLY_SAMPLE_CUT_OFF = True
LIMIT_SAMPLES = 50 # sample cells
LIMIT_FEATURES = 500 # sample genes
REPETITIONS = 10 # repetitions
K = 5 # External fold iteration
L = 3 # Internal fold iterations
VERBOSE = True

DATASET_PATH = '../data_preprocessed/everything_cleaned_normalized_reduced.csv'

#
# Features
#

def trainAllAlgorithms(nCV):
    result = nCV.trainAllAlgorithmsCV(REPETITIONS)

    nCV.printMetricStatistics()

    best_algorithm, median_scores = nCV.findBestAlgorithm()

    print(median_scores)

    print("Winner is:", best_algorithm)

    best, report = nCV.trainBestModel(best_algorithm)

    print(best)

    print(report)


def trainOneAlgorithm(nCV, algorithm):
    nCV.initializeMetricStatistics()

    nCV.trainSpecificAlgorithmCV(algorithm, REPETITIONS)

    nCV.printMetricStatistics(algorithm)

    best_algorithm, median_scores = nCV.findBestModelForAlgorithm(algorithm)

    best, report = nCV.trainBestModel(best_algorithm)

    return best, report


def main():

    print("Loading dataset ... ")

    dataset = pd.read_csv(DATASET_PATH,index_col = 0)

    # ----------------------
    # for dev only:
    # ----------------------

    if APPLY_SAMPLE_CUT_OFF:
        print("Applying cut off: ", LIMIT_SAMPLES, LIMIT_FEATURES)
        print("Before cut off: ", dataset.shape)
        dataset = dataset.iloc[0:LIMIT_SAMPLES, dataset.shape[1]-LIMIT_FEATURES:]
        print("After cut off: ", dataset.shape)

    distinct_count = dataset['Classification'].nunique()

    print("Dataset shape before training:", dataset.shape)
    print("Total classes detected: ", distinct_count)

    classifiers = {
        'Logistic Regression': {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.1, 1, 10]
        },
        'Random Forest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30, 40]
        },
        'k-Nearest Neighbors': {
            'n_neighbors': [3, 5, 10],
            'weights': ['uniform', 'distance', None]
        },
        'Linear Discriminant Analysis': {
            'tol': [1e-10, 1e-9, 1e-8],
            'solver': ['svd', 'lsqr', 'eigen']
        },
        'Support Vector Machines': {
            'kernel': ['poly', 'linear', 'rbf'],
            'C': [0.5, 1, 1.5, 2]
        }
    }

    metrics = [
        'Matthews Correlation Coefficient',
        'Balanced Accuracy',
        'F1 Score',
        'F2 Score',
        'Sensitivity (Recall)',
        'Precision'
    ]


    nCV = NestedCrossValidation(dataset, classifiers, K, L, metrics, VERBOSE)

    nCV.printInfo()


    #
    # All algorithms at onece:
    #
    # trainAllAlgorithms(nCV)

    #
    # Train specific algorithm
    #

    best, report = trainOneAlgorithm(nCV, 'Logistic Regression')

    joblib.dump(best,  '../models/logistic_regression_model.pkl')

    print(best)

    print(type(best))

    print(report)






    # best, report = trainOneAlgorithm(nCV, 'Random Forest')


    # for al in classifiers.keys():
    #     nCV.printModels(al)


    # nCV.plotMetric("Matthews Correlation Coefficient")
    #
    # # nCV.plotMetricDataForAlgorithm("Logistic Regression")
    #


main()

