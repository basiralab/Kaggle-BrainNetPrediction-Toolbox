from sklearn.model_selection import KFold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr
import pandas as pd
import random as r
r.seed(1)
np.random.seed(1)
import warnings
warnings.filterwarnings('ignore')


def load_data(csv):

    """
    The method reads train and test data from their dataset files.
    Then, it splits train data into features and labels.
    Parameters
    ----------
    train_file: directory of the file in which train data set is located
    test_file: directory of the file in which test data set is located
    """
    data = genfromtxt(csv, delimiter=',')[1:, 1:]  # ignoring first row and column since they are not data
    return data

def preprocessing(data): 
    
    """
    Takes outliers in the data and converts them into mean of the data
    ----------
    data: input data to process
    """
    std = np.std(data, axis=0)  # standard deviation
    mean = np.mean(data, axis=0)  # mean
    lower_limit = mean - std * 3  # if a data point is lower than this, it is an outlier
    upper_limit = mean + std * 3  # if a data point is higher than this, it is an outlier

    for i in data:
        if np.shape(np.argwhere(i > upper_limit)) != (0, 1):  # if there is points which are higher than upper limit
            i[np.argwhere(i > upper_limit)] = mean[np.argwhere(i > upper_limit)]  # replace them with mean of the data

        if np.shape(np.argwhere(i < lower_limit)) != (0, 1):  # if there is points which are lower than lower limit
            i[np.argwhere(i < lower_limit)] = mean[np.argwhere(i < lower_limit)]  # replace them with mean of the data

    return data  # return the cleaned data


def train_model(x, y, num_of_iters=10000, learning_rate=0.1):

    """
    Trains model using gradient descent algorithm
    Parameters
    ----------
    x: features of train data
    y: targets of train data
    num_of_iters: number of iterations
    learning_rate: learning rate
    """
    d = np.shape(x)[1]  # number of features in training set
    n = np.shape(x)[0]  # number of data in training set
    # initializing w and b (w is random and b iz zero)
    w_ = np.random.rand(1, d)
    b_ = np.zeros((1, d))
    for i in range(num_of_iters):
        y_predicted = w_ * x + b_  # prediction done with current w and b
        dw = (-2 / n) * sum(x * (y - y_predicted))  # Derivative with respect to w
        db = (-2 / n) * sum(y - y_predicted)  # Derivative ith respect to b
        w_ = w_ - learning_rate * dw  # updating w
        b_ = b_ - learning_rate * db  # updating b
    return w_, b_  # returning optimal parameters that are found as a result of gradient descent


def predict(x, w, b): 

    """
    The method predicts labels for testing data samples.
    Parameters
    ----------
    model: trained learning model (KNN)
    x_test: features of testing data
    """
    return w * x + b

def cv5(data_t0, data_t1, num_of_iterations, l_rate):

    """
    The method applies 5 fold cross validation
    Parameters
    ----------
    data_t0: x
    data_t1: y
    num_of_iterations: number of iterations
    l_rate: learning rate
    """
    kf = KFold(n_splits=5, shuffle = True, random_state=1)
    mse_errors = []
    mae_errors = []
    pearsons = []
    ws = []  # this will store 5 w's
    bs = []  # this will store 5 b's
    for trainIndex, testIndex in kf.split(data_t0):
        train_t0, test_t0 = data_t0[trainIndex], data_t0[testIndex] #Split Data into train and test sets
        train_t1, test_t1 = data_t1[trainIndex], data_t1[testIndex]
        train_t1 = preprocessing(train_t1)  # cleaning outliers
        w, b = train_model(train_t0, train_t1, num_of_iterations, l_rate)  # producing w and b of a fold
        y_train_predicted = predict(test_t0, w, b)  # predicting output in a fold
        mse_errors.append( mean_squared_error(y_train_predicted, test_t1)) # calculating error in the fold and adding it into the array
        mae_errors.append( mean_absolute_error(y_train_predicted, test_t1) )
        pearsons.append( pearsonr(y_train_predicted.flatten(), test_t1.flatten())[0] )
        ws.append(w)  # adding obtained w into the array
        bs.append(b)  # adding obtained b into the array
        """
        for i in range(y_train_predicted.shape[0]):
            print(mean_absolute_error(y_train_predicted[i,:], test_t1[i,:] ))
        print("----")
        for i in range(y_train_predicted.shape[0]):
            print(pearsonr(y_train_predicted[i,:], test_t1[i,:])[0])
        """
    
    print("mses: ", mse_errors)
    print("maes: ", mae_errors)
    print("pears", pearsons)
    
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
    predictions = np.ravel(predictions, order='C')
    np.savetxt(filename+".csv", np.dstack((np.arange(0, predictions.size), predictions))[0], "%d,%f", header="ID,predicted")
    print(filename+".csv saved")

x_train = load_data('train_t0.csv')
y_train = load_data('train_t1.csv')
x_test = load_data('test_t0.csv')

number_of_iterations = 10000  # number of iterations which will be used in gradient descent algorithm
learning_rate = 0.1  # learning rate which will be used in gradient descent algorithm

y_train = preprocessing(y_train)  # cleaning outliers
w, b = train_model(x_train, y_train, number_of_iterations, learning_rate)  # obtaining w and b by using gradient

predictions = predict(x_test, w, b)  # testing the found parameters (w and b)
write_output("team13", predictions)

print("--Starting to 5 fold cross validation--")
x_train = load_data('train_t0.csv')
y_train = load_data('train_t1.csv')
x_test = load_data('test_t0.csv')
cv5(x_train, y_train, number_of_iterations, learning_rate)  # applying 5 fold cross validation
print("--5 fold cross validation is over--")