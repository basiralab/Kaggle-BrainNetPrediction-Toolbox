import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as r
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_data(csv):

    """
    The method reads data from their dataset files.
    Parameters
    ----------
    csv: directory of the file in which data set is located
    """
    dataset = pd.read_csv(csv)
    del dataset['ID']
    return dataset

def removeEmpty(X, test):
    
    """
    The method removes empty columns from dataset files.
    Parameters
    ----------
    X, test: directory of the file in which data set is located
    """
    X = X.loc[:, (X != 0).any(axis=0)]
    test = test.loc[:, (test != 0).any(axis=0)]
    return X, test

def removeDuplicates(X, test):
            
    """
    The method removes the duplicates
    Parameters
    ----------
    X: train data
    test: test data1
    """
    column_count = len(X.columns)
    dp_indexes = []
    for i in range(0, column_count - 1):
      a = X.iloc[:, i]
      for j in range(i + 1, column_count):
        b = X.iloc[:, j]
        if np.array_equal(a, b):
          dp_indexes.append(j)
    dp_indexes = list(set(dp_indexes))
    X.drop(X.columns[dp_indexes], axis=1, inplace=True) 
    test.drop(test.columns[dp_indexes], axis=1, inplace=True) 
    return X, test

def dim_reduction(X, test):
        
    """
    The method reduces the dimension
    Parameters
    ----------
    X: train data
    test: test data1
    """
    pca = PCA(n_components = 11)
    X = pca.fit_transform(X)
    test = pca.transform(test)
    X = pd.DataFrame.from_records(X)
    test = pd.DataFrame.from_records(test)
    return X, test

def preprocessing(train_t0, test_t0):
    
    """
    The method removes empty and duplicates, reduces the dimension
    Parameters
    ----------
    train_t0: x
    train_t1: y
    """
    train_t0, test_t0 = removeEmpty(train_t0, test_t0)
    train_t0, test_t0 = removeDuplicates(train_t0, test_t0)
    train_t0, test_t0 = dim_reduction(train_t0, test_t0)
    return train_t0, test_t0

def train_model(train_t0, train_t1):

    """
    The method creates a learning model and trains it by using training data.
    Parameters
    ----------
    train_t0: x
    train_t1: y
    """
    models = []
    regressor = SVR(kernel='rbf', C=0.032, gamma='scale', epsilon=.022, tol=0.001)
    for i in range(0,595):
        models.append(regressor.fit(train_t0, train_t1.iloc[:, i]))
    return models

def predict(test_t0, models):
   
    """
    The method predicts for testing data samples by using trained learning model.
    Parameters
    ----------
    test_t0: features of testing data
    models: trained learning model
    """
    preds = []
    for i in range(0,595):
        pred = models[i].predict(test_t0)
        preds.append(pred)
    return np.asarray(preds)

def cv5(X, Y):

    """
    Applies 5 fold cross validation on given dataset
    Parameters
    ----------
    x: train_t0
    y: train_t1
    """
    kf = KFold(n_splits=5, shuffle = True, random_state=1)
    mse_errors = []
    mae_errors = []
    pearsons = []
    ws = []  # this will store 5 w's
    bs = []  # this will store 5 b's
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        X_train, X_test = preprocessing(X_train, X_test)
        model = train_model(X_train, Y_train)
        preds = predict(X_test, model)
        for i, pred in enumerate(preds):
            mse_errors.append( mean_squared_error(pred, Y_test.iloc[:, i])) # calculating error in the fold and adding it into the array
            mae_errors.append( mean_absolute_error(pred, Y_test.iloc[:, i]) )
            pearsons.append( pearsonr(pred.flatten(), Y_test.iloc[:, i].values.flatten())[0] )
    
    print("Average error of five fold cross validation MSE:", np.sum(mse_errors) / 5)
    print("Average error of five fold cross validation MAE:", np.sum(mae_errors) / 5)
    print("Average error of five fold cross validation pearson:", np.sum(pearsons) / 5)

    print(" std of five fold cross validation MSE:", np.std(mse_errors))
    print(" std of five fold cross validation MAE:", np.std(mae_errors))
    print(" std of five fold cross validation pearson:", np.std(pearsons))

def write_output(filename, predictions):

    """
    Writes predictions as desired in the submission process
    Parameters
    ----------
    filename: file path for saving file
    predictions: model outputs
    """
    meltedDF = np.asarray(predictions).flatten()
    with open(filename+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Predicted"])
        for i in range(0,47600):
            if(meltedDF[i] < 10 ** (-50)):
                meltedDF[i] = 0
            writer.writerow([i, meltedDF[i]])

train_t0 = load_data("train_t0.csv")
train_t1 = load_data("train_t1.csv")
test_t0 = load_data("test_t0.csv")

train_t0_clean, test_t0_clean = preprocessing(train_t0, test_t0)
model = train_model(train_t0_clean, train_t1)
preds = predict(test_t0_clean, model)

write_output('results_team_20', preds)
cv5(train_t0, train_t1)  # applying 5 fold cross validation
