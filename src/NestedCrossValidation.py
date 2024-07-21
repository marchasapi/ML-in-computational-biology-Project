import random
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score, fbeta_score, \
    precision_recall_curve, confusion_matrix, precision_score, average_precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
import statistics

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

class NestedCrossValidation:
    def __init__(self, dataset : DataFrame, classifiers : dict, K : int, L : int, metrics: list, verbose : bool) -> None:
        super().__init__()
        self.dataset = dataset
        self.classifiers = classifiers
        self.K = K
        self.L = L
        self.metrics = metrics
        self.metricStatistics = {}
        self.models = {}
        self.verbose = verbose

    def printInfo(self):
        print(f"NestedCrossValidation initialized:")
        print(f"  - Dataset shape    : {self.dataset.shape}")
        print(f"  - Total classifiers: {len(self.classifiers)}")
        print(f"  - Classifiers      : ", self.classifiers)
        print(f"  - Outer loops      : ", self.K)
        print(f"  - Inner loops      : ", self.L)
        print(f"  - Metrics          : ", self.metrics)

    def printModels(self, algorithm):
        for m in self.models[algorithm]:
            print(m)

    def calculateMetrics(self, algorithm, best, X_train, y_train, X_test, y_test, y_pred):
       #self.metricStatistics[algorithm]["Matthews Correlation Coefficient"].append(matthews_corrcoef(y_test, y_pred))
        self.metricStatistics[algorithm]["Balanced Accuracy"].append(balanced_accuracy_score(y_test, y_pred))
        self.metricStatistics[algorithm]["F1 Score"].append(f1_score(y_test, y_pred, average='weighted'))
       #self.metricStatistics[algorithm]["F2 Score"].append(fbeta_score(y_test, y_pred, beta=2, average='weighted'))
        self.metricStatistics[algorithm]["Sensitivity (Recall)"].append(recall_score(y_test, y_pred, average='weighted'))
        self.metricStatistics[algorithm]["Precision"].append(precision_score(y_test, y_pred, average='weighted'))

        if algorithm == 'Support Vector Machine':
            calibrated_svm = CalibratedClassifierCV(best, method='sigmoid', cv='prefit')
            calibrated_svm.fit(X_train, y_train)
        else:
            y_proba = best.predict_proba(X_test)[:, 1]


    def trainAlgorithm(self, iteration, algorithm, inner_loop, X_train, y_train, X_test, y_test, parameters, calculateMetrics):
        best = None

        if (algorithm == 'Logistic Regression'):
            model = LogisticRegression()
            LR_grid = GridSearchCV(model, param_grid=parameters, scoring='accuracy', cv=inner_loop, refit=True)
            result = LR_grid.fit(X_train, y_train)
            best = result.best_estimator_
            y_pred = best.predict(X_test)

        if (algorithm == 'Random Forest'):
            model = RandomForestClassifier(n_estimators=100)
            GNB_grid = GridSearchCV(model, cv=inner_loop, param_grid=parameters, scoring='accuracy', refit=True)
            result = GNB_grid.fit(X_train, y_train)
            best = result.best_estimator_
            y_pred = best.predict(X_test)

        if (algorithm == 'k-Nearest Neighbors'):
            model = KNeighborsClassifier()
            kNN_grid = GridSearchCV(model, cv=inner_loop, param_grid=parameters, scoring='accuracy', refit=True)
            result = kNN_grid.fit(X_train, y_train)
            best = result.best_estimator_
            y_pred = best.predict(X_test)

        if (algorithm == 'Linear Discriminant Analysis'):
            model = LinearDiscriminantAnalysis()
            LDA_grid = GridSearchCV(model, cv=inner_loop, param_grid=parameters, scoring='accuracy', refit=True)
            result = LDA_grid.fit(X_train, y_train)
            best = result.best_estimator_
            y_pred = best.predict(X_test)

        if (algorithm == 'Support Vector Machines'):
            model = SVC(probability=True)
            SVM_grid = GridSearchCV(model, cv=inner_loop, param_grid=parameters, scoring='accuracy', refit=True)
            result = SVM_grid.fit(X_train, y_train)
            best = result.best_estimator_
            y_pred = best.predict(X_test)

        if calculateMetrics:
            self.models[algorithm].append(best)

            if best != None:
                self.calculateMetrics(algorithm, best, X_train, y_train, X_test, y_test, y_pred)
            else:
                exit(1)

        report = classification_report(y_test, y_pred)

        return best, report

    def initializeMetricStatistics(self):
        self.metricStatistics = {}
        self.models = {}
        for algorithm in self.classifiers.keys():
            self.metricStatistics[algorithm] = {}
            self.models[algorithm] = list()
            for metric in self.metrics:
                self.metricStatistics[algorithm][metric] = list()

    def printMetricStatistics(self, filter_algorithm = None):
        print("{:<40}".format("Metric"), end='')
        if filter_algorithm == None:
            for algorithm in self.classifiers:
                print("{:<30}".format(algorithm), end='')
        else:
            print("{:<30}".format(filter_algorithm), end='')

        print()

        for metric in self.metrics:
            print("{:<40}".format(metric), end='')
            if filter_algorithm == None:
                for algorithm in self.classifiers.keys():
                    if (len(self.metricStatistics[algorithm][metric]) > 0):
                        print("{:<30}".format(statistics.median(self.metricStatistics[algorithm][metric])), end='')
                    else:
                        print("{:<30}".format(0), end='')
            else:
                if (len(self.metricStatistics[filter_algorithm][metric]) > 0):
                    print("{:<30}".format(statistics.median(self.metricStatistics[filter_algorithm][metric])), end='')
                else:
                    print("{:<30}".format(0), end='')

            print()

    def trainSpecificAlgorithmCV(self, algorithm, repetitions : int, limit_loops = None):
        X = self.dataset.iloc[:, :-1]
        Y = self.dataset.iloc[:, -1]

        print(X.shape)
        print(Y.shape)

        cv_outer = StratifiedKFold(n_splits=self.K, shuffle=True)
        inner_loop = StratifiedKFold(n_splits=self.L, shuffle=True)

        counter = 0

        for i in range(repetitions + 1):
            fold = 1
            for train_ix, test_ix in cv_outer.split(X, Y): # K splits
                print(algorithm, " - Repetition: ", i + 1, " Fold: " , fold)
                X_train = X.iloc[train_ix]
                X_test = X.iloc[test_ix]
                y_train = Y[train_ix]
                y_test = Y[test_ix]

                self.trainAlgorithm(i + 1, algorithm, inner_loop, X_train, y_train, X_test, y_test, self.classifiers[algorithm], True)
                fold = fold + 1

            counter = counter + 1
            if limit_loops != None and counter >= limit_loops:
                return None

        return None

    def trainAllAlgorithmsCV(self, repetitions : int, limit_loops = None):
        self.initializeMetricStatistics()

        X = self.dataset.iloc[:, :-1] 
        Y = self.dataset.iloc[:, -1] 

        print(X.shape)
        print(Y.shape)

        cv_outer = StratifiedKFold(n_splits=self.K, shuffle=True)
        inner_loop = StratifiedKFold(n_splits=self.L, shuffle=True)

        counter = 0

        for algorithm in self.classifiers.keys():
            for i in range(repetitions + 1):
                fold = 1
                for train_ix, test_ix in cv_outer.split(X, Y): # K splits
                    print(algorithm, " - Repetition: ", i + 1, " Fold: " , fold)
                    X_train = X.iloc[train_ix]
                    X_test = X.iloc[test_ix]
                    y_train = Y[train_ix]
                    y_test = Y[test_ix]

                    self.trainAlgorithm(i + 1, algorithm, inner_loop, X_train, y_train, X_test, y_test, self.classifiers[algorithm], True)
                    fold = fold + 1

                counter = counter + 1
                if limit_loops != None and counter >= limit_loops:
                    return None

        return None

    def findBestAlgorithm(self):
        median_scores = {}

        for algorithm in self.classifiers.keys():
            temp = []

            for metric, values in self.metricStatistics[algorithm].items():
                if len(values) == 0:
                    break;
                temp.append(statistics.median(values))

            if len(temp) == 0:
                break;
            median_scores[algorithm] = statistics.median(temp)

        best_algorithm = max(median_scores, key=median_scores.get)

        return best_algorithm, median_scores

    def findBestModelForAlgorithm(self, algorithm):
        median_scores = {}

        temp = []

        for metric, values in self.metricStatistics[algorithm].items():
            if len(values) == 0:
                break;
            temp.append(statistics.median(values))

        median_scores[algorithm] = statistics.median(temp)

        best_algorithm = max(median_scores, key=median_scores.get)

        return best_algorithm, median_scores

    def trainBestModel(self, algorithm):
        X = self.dataset.iloc[:, :-1] # fix
        Y = self.dataset.iloc[:, -1] # fix

        inner_loop = StratifiedKFold(n_splits=5, shuffle=True)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True)

        parameters = self.classifiers[algorithm]

        best, report = self.trainAlgorithm(0, algorithm, inner_loop, X_train, y_train, X_test, y_test, parameters, False)

        return best, report

    def getBoxPlotData(self, metric):
        data = []

        for algorithm in self.classifiers.keys():
            data.append(self.metricStatistics[algorithm][metric])

        return data
    
    def plotMetric(self, metric, algorithm=None):
        if algorithm:
            # Check if the algorithm exists in the classifiers
            if algorithm not in self.classifiers:
                print(f"Algorithm {algorithm} not found.")
                return

            # Get the data for the specific algorithm
            data = [self.metricStatistics[algorithm][metric]]
            algorithms = [algorithm]
        else:
            # Get the data for all algorithms
            data = self.getBoxPlotData(metric)
            algorithms = self.classifiers.keys()

        # Check for empty data lists
        for i, d in enumerate(data):
            if len(d) == 0:
                print(f"No data for algorithm: {algorithms[i]}")
                return

        plt.boxplot(data, vert=True, patch_artist=True, labels=(algorithms))
        plt.title(metric)
        plt.xticks(rotation=45)
        plt.ylabel("metric")

        medians = [np.median(d) for d in data]
        for i, median in enumerate(medians):
            plt.text(i + 1, median, f"{median:.2f}", horizontalalignment='center', verticalalignment='bottom',
                    fontsize=10)

        for i, d in enumerate(data):
            q1 = np.percentile(d, 25)
            q3 = np.percentile(d, 75)
            iqr = q3 - q1
            plt.text(i + 1, q3, f"IQR: {iqr:.2f}", horizontalalignment='center', verticalalignment='bottom',
                    fontsize=10)

        plt.show()




    def getXyData(self, algorithm):
        return self.metricStatistics[algorithm]

    def plotMetricDataForAlgorithm(self, algorithm):
        data = self.getXyData(algorithm)
        labels = list(data.keys())
        values = list(data.values())

        plt.figure(figsize=(10, 6))

        for label, value in zip(labels, values):
            plt.plot(value, label=label, marker='o', markersize=3)

        plt.title('Performance Metrics for ' + algorithm)
        plt.xlabel('Iteration')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()
