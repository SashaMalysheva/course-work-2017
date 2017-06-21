import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import autosklearn.classification


def get_data():
    """
    Get a random subset of 1000 samples from the USPS digit recognition dataset.
    """
    digits = sklearn.datasets.load_linnerud()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(X, y, random_state=1)
    return X_train, X_test, y_train, y_test


def create_models(X_train, y_train, d=10):
    """
    With auto_sklearn library create d models,
    on different parts of data train.
    :param X_train: 2d array
        Train data set feats.
    :param y_train: array
        Train data set labels.
    :param d: int, default = 10
        Number of models.
    :return: list of models.
    """
    n = len(y_train)/d
    automl_models = []
    for i in range(d):
        X_i = X_train[i * n: (i+1) * n]
        y_i = y_train[i * n: (i+1) * n]
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
        automl.fit(X_i, y_i)
        #print(automl.show_models())
        automl_models.append(automl)
    return automl_models


def generate_table(X_train, y_train, automl_models=[]):
    """
    Generate table:
        T[i, j] - True, if model[i] correctly predict y[j]
    :param X_train: 2d array
        Train data set feats.
    :param y_train: array
        Train data set labels.
    :param automl_models: list
        List of fitted machine learning models.
    :return: np.array
    """
    T = np.zeros((len(y_train), len(automl_models)))
    for i in range(len(automl_models)):
        predictions = automl_models[i].predict(X_train)
        for j in range(len(y_train)):
            T[j, i] = 1 if predictions[i] == y_train[i] else 0
    return T


def predict_weights(X_train, X_test, T):
    """
    Generate table of weights
         weights[i, j] - The probability that the model[i] predicts correctly y[j]
    :param X_train: 2d array
        Train data set feats.
    :param X_test: 2d array
        Test data set feats.
    :param table: np.array
        T[i, j] - True, if model[i] correctly predict y[j]
    :return: np.array
    """
    regr = DecisionTreeRegressor(max_depth=5)
    regr.fit(X_train, T)
    return regr.predict(X_test)


def votingClassifier(weights, automl_models, X_test):
    """
    Generate prediction of test data set labels.
    :param weights: np.array
        weights[i, j] - The probability that the model[i] predicts correctly y[j]
    :param automl_models: list
        list of models.
    :param X_test:
        Test data set feats.
    :return: np.array
    """
    predictions = np.zeros(len(X_test))
    answers = np.zeros((len(automl_models), len(X_test)))
    for i in range(len(automl_models)):
        answers[i] = automl_models[i].predict(X_test)

    for j in range(len(X_test)):
        (values, counts) = np.unique(answers[:,j], return_counts=True)
        ind = np.argmax(counts)
        predictions[j] = values[ind]
    return predictions


def main():
    """
    Generate and run d auto_ml models on train data set,
    And initialize a VotingClassifier on test data set.
    Finally print Accuracy score.
    """
    X_train, X_test, y_train, y_test = get_data()
    automl_models = create_models(X_train, y_train)
    T = generate_table(X_train, y_train, automl_models)
    weights = predict_weights(X_train, X_test, T)
    predictions = votingClassifier(weights, automl_models, X_test, X_train, y_train)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
