"""
Target Problem:
---------------
* Predict the evolution of brain connectivity over time.

Proposed Solution (Machine Learning Pipeline):
----------------------------------------------
* TruncatedSVD -> RandomForest, KMeansClustering

Input to Proposed Solution:
---------------------------
* Directories of training data and labels(measurements at two different timepoints), and testing data in csv file format.
* These data should be stored in n x m pattern in csv file format.

  Typical Example:
  ----------------
  n x m samples in train_t0 csv file (n number of samples, m number of features)
  n x m samples in train_t1 csv file (n number of samples, m number of features)
  k x m samples in test_t0 csv file (k number of samples, m number of features)

* These data set files are ready by load_data() function.

Output of Proposed Solution:
----------------------------
* Predictions generated by learning model for testing set
* They are stored in "submission.csv" file.

Code Owner:
-----------
* Copyright © Team 9. All rights reserved.
* Copyright © Istanbul Technical University, Learning From Data Fall 2021. All rights reserved.
"""

import models
import sklearn
import numpy as np
import pandas as pd
import statistics
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr as pr
np.random.seed(1)

def load_data(files):
    
    """
    The method reads train and test data from their dataset files.
    Then, it splits train data into features and labels.
    Parameters
    ----------
    train_file: directory of the file in which train data set is located
    test_file: directory of the file in which test data set is located
    """

    def get_feature_matrix(frame):
        '''Given a dataframe, returns the corresponding matrix

        Args:
            frame (pd.DataFrame): Dataframe to be extracted as feature matrix

        Returns:
            (np.array): Feature matrix of shape (n_subjects, 595)
        '''
        return frame.drop(columns="ID").to_numpy()

    train_data = get_feature_matrix(pd.read_csv(files[0]))
    train_label = get_feature_matrix(pd.read_csv(files[1]))
    test_data = get_feature_matrix(pd.read_csv(files[2]))
    return train_data, train_label, test_data

def train(x_tra, y_tra):
    
    """
    The method creates a learning model and trains it by using training data.
    Parameters
    ----------
    x_tra: features of training data
    y_tra: labels of training data
    x_tst: features of testing data
    """

    regressors = (
        models.FeatureClusterRF(n_selected_features=250, n_clusters=5),
        models.BallTreeWeighted(n_components=55, k=25),
    )

    model = models.MeanEnsembler(regressors, [75, 25])

    return model.fit(x_tra, y_tra)

def predict(x_tst, model):
    
    """
    The method predicts labels for testing data samples by using trained learning model.
    Parameters
    ----------
    x_tst: features of testing data
    model: trained learning model
    """

    return model.predict(x_tst)

def cv(x_tra, y_tra):
    """
    Performs 5-fold crossvalidation.
    Parameters
    ----------
    x_tra: features of testing data
    y_tra: labels of testing data
    """

    regressors = (
        models.FeatureClusterRF(n_selected_features=250, n_clusters=5),
        models.BallTreeWeighted(n_components=55, k=25),
    )

    estimator = models.MeanEnsembler(regressors, [75, 25])

    evaluator = lambda scores, preds: {'mse': sklearn.metrics.mean_squared_error(scores, preds)}

    kfold = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=1)
    fold_results = []
    overall_preds = []
    msqr = []
    mabs = []
    pearson = []
    i = 1
    df_mae = pd.DataFrame()
    df_pcc = pd.DataFrame()
    for train_idx, test_idx in kfold.split(x_tra):
        X_train, X_test = x_tra[train_idx], x_tra[test_idx]
        Y_train, Y_test = y_tra[train_idx], y_tra[test_idx]
        estimator.fit(X_train, Y_train)
        Y_pred = estimator.predict(X_test)
        result = evaluator(Y_pred, Y_test)
        fold_results.append(result)
        overall_preds.extend(Y_pred)
        for ind, row in zip(range(30), range((i-1)*Y_test.shape[0], i*Y_test.shape[0])):
            df_mae[row] = [mae(Y_pred[ind, :].flatten(), Y_test[ind, :].flatten())]
            df_pcc[row] = [pr(Y_pred[ind, :].flatten(), Y_test[ind, :].flatten())[0]]
        msqr.append(mse(Y_pred.flatten(), Y_test.flatten()))
        mabs.append(mae(Y_pred.flatten(), Y_test.flatten()))
        pearson.append(pr(Y_pred.flatten(), Y_test.flatten()))
        i += 1
    df_mae.to_csv("9_mae.csv")
    df_pcc.to_csv("9_pcc.csv")
    print("MAE:")
    for i,stat in enumerate(mabs):
        print(i+1, stat)
    print("PCC:")
    for i,stat in enumerate(pearson):
        print(i+1, stat)

    print(statistics.mean(msqr), statistics.stdev(msqr))
    print(statistics.mean(mabs), statistics.stdev(mabs))
    print(statistics.mean(i[0] for i in pearson), statistics.stdev(i[0] for i in pearson))
    return fold_results

def write_output(filename, predictions):

    """
    Writes the outputted predictions to a file.
    Parameters
    ----------
    filename: file path for saving file
    predictions: numpy array object containing predictions
    """
    predictions = predictions.flatten()
    submission = pd.DataFrame(predictions, columns = ["Predicted"])
    submission.index.name = "ID"
    submission.to_csv(filename)

if __name__ == "__main__":
    filenames = ["train_t0.csv", "train_t1.csv", "test_t0.csv"]
    x_tra, y_tra, x_tst = load_data(filenames)
    cv(x_tra, y_tra)
    model = train(x_tra, y_tra)
    predictions = predict(x_tst, model)
    write_output("submission.csv", predictions)
