"""
Target Problem:
---------------
* To train a model to predict the brain connectivity for the next time point given the brain connectivity at current time point.
Proposed Solution (Machine Learning Pipeline):
----------------------------------------------
* Preprocessing Method (if any) -> Dimensionality Reduction method (if any) -> Learner
Input to Proposed Solution:
---------------------------
* Directories of training and testing data in csv file format
* These two types of data should be stored in n x m pattern in csv file format.
  Typical Example:
  ----------------
  n x m samples in training csv file (Explain n and m) 
  k x s samples in testing csv file (Explain k and s

Output of Proposed Solution:
----------------------------
* Predictions generated by learning model for testing set
* They are stored in "submission.csv" file. (Change the name file if needed)
Code Owner:
-----------
* Copyright © Team X. All rights reserved.
* Copyright © Istanbul Technical University, Learning From Data Spring/Fall 2020. All rights reserved. 
"""


from sklearn import random_projection, preprocessing

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import ElasticNet, OrthogonalMatchingPursuit, SGDRegressor, Lars, BayesianRidge
from sklearn.linear_model import ARDRegression, PassiveAggressiveRegressor, RANSACRegressor
from sklearn.linear_model import Ridge, SGDRegressor, TheilSenRegressor, HuberRegressor

from sklearn.feature_selection import GenericUnivariateSelect,SelectFromModel, VarianceThreshold,  RFECV, RFE
from sklearn.svm import SVC, SVR, NuSVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error as mae
from scipy.stats.stats import pearsonr
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random as r
r.seed(1)
np.random.seed(1)
import warnings
warnings.filterwarnings("ignore")

def load_data(x_train, y_train, x_test):
    
    """
    Parameters
    ----------
    x_train: directory of training data t0
    y_train: directory of training data t1
    x_test: directory of test data t0
    
    """
    
    train_t0 = pd.read_csv(x_train).loc[:,"f1":]
    train_t1 = pd.read_csv(y_train).loc[:,"f1":]
    test_t0 = pd.read_csv(x_test).loc[:,"f1":]
    return train_t0, train_t1, test_t0


def preprocessing(x_tra, y_tra, x_tst):

    """
    * Explain the method
    * Explain which function does it
    ----------
    x_tra: features of training data
    y_tra: labels of training data
    x_tst: features of test data
    """
    transformer = GenericUnivariateSelect(score_func=lambda X, y: X.mean(axis=0), mode='percentile', param=85)
    X_train_new = transformer.fit_transform(x_tra, y_tra)
    X_test_new = transformer.transform(x_tst)
    
    transformer = PCA(n_components=21)
    X_train_new = transformer.fit_transform(X_train_new)
    X_test_new = transformer.transform(X_test_new)
    
    return X_train_new, X_test_new

def train_voting(X,y):
    
    
    """
    The method creates a learning model and trains it by using training data.
    Parameters
    ----------
    X: preprocessed features of training data
    y: features of training data
    
    """
    
    # X after preprocessing
    models = [
          MultiOutputRegressor(AdaBoostRegressor(n_estimators=100, learning_rate=0.04)),
          KNeighborsRegressor(algorithm='ball_tree', n_neighbors=24),
          Lasso(alpha=0.001, tol=1e-10, max_iter=10000),
          KNeighborsRegressor(n_neighbors=27, algorithm='kd_tree', weights='distance', leaf_size=15),
          MultiOutputRegressor(BayesianRidge(tol=1e-2, n_iter=15)),
          KNeighborsRegressor(n_neighbors=15, algorithm='brute', weights='distance', leaf_size=15),
          MultiOutputRegressor(BayesianRidge(tol=1e-2, n_iter=50)),
          KNeighborsRegressor(n_neighbors=35, algorithm='brute', weights='distance', leaf_size=15),
          OrthogonalMatchingPursuit(),
          MultiOutputRegressor(LGBMRegressor(objective='regression')),
          MultiOutputRegressor(xgb.XGBRegressor()),
          MultiOutputRegressor(BayesianRidge(tol=1e-2, n_iter=100)),
          KNeighborsRegressor(algorithm='ball_tree', n_neighbors=24),
          KNeighborsRegressor(n_neighbors=27, algorithm='kd_tree', weights='distance', leaf_size=15),
          KNeighborsRegressor(n_neighbors=15, algorithm='brute', weights='distance', leaf_size=15)
    ]
    for model in models:
        model.fit(X,y)
    return models

def voting_predict(X, models):
    """
    Parameters
    ----------
    X: preprocessed features of test data
    models: trained models of voting algorithms
    
    """
    pred=0
    for model in models:
        pred+=model.predict(X)
    return pred/len(models)

def train_feature_based(X,y,sel_models):
    
    """
    Parameters
    ----------
    X: features of training data
    y: features of training data
    
    sel_models: selected models 
        We selected the best machine learning algorithms for each feature 
        and saved it to Selected_models.pickle file.
    
    """
    
    
    trained_models = []
    for i in range(1,596):
        trained = sel_models[i-1].fit(X["f"+str(i)].values.reshape(-1, 1),y["f"+str(i)].values)
        trained_models.append( trained )
    return trained_models

def predict_feature_based(X,trained_best_models):
    
    """
    Parameters
    ----------
    X: features of test data
    
    trained_best_models: 595 different trained model.
        We selected the best machine learning algorithms for each feature 
        and trained these models with train_selected_models function.
    """
    
    preds = []
    for i in range(0,595):
        prediction = trained_best_models[i].predict(X["f"+str(i+1)].values.reshape(-1, 1))
        preds.append(prediction)
        
    np_pred = np.array(preds).T
    return np_pred

def write_output(filename, prediction):
    
    """
    Parameters
    ----------
    filename: name of the csv file
    prediction: prediction
    
    """
    
    sample_submission = pd.read_csv("sampleSubmission.csv")
    sample_submission["predicted"] = prediction.flatten()
    sample_submission.to_csv(filename+".csv", index=False)

def cv5(train_X,train_y):
    column_names=[]
    for i in range(1,596):
        column_names.append("f"+str(i))
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    Five_fold_predictions = []
    Mse_results = []
    Mae_results = []
    test_indexes = []

    final_prediction = np.zeros(train_X.shape[0]*595).reshape(train_X.shape[0],595)

    X = train_X.values
    y = train_y.values

    counter=0
    for train_index, test_index in kf.split(X):
        counter+=1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        test_indexes.append(test_index)
        #print("Mse of Cross Validation "+str(counter)+":")
        x_train_red, x_test_red  =  preprocessing(X_train, y_train, X_test)
        # Train voting models
        voting_models = train_voting(x_train_red, y_train)
        
        
        df_X_train = pd.DataFrame(data=X_train, columns=column_names)
        df_Y_train = pd.DataFrame(data=y_train, columns=column_names)
        df_X_test = pd.DataFrame(data=X_test, columns=column_names)

        trained_best_models = train_feature_based(df_X_train, df_Y_train, selected_models)
        
        # We combined these two models to increase the accuracy
        pred_voting = voting_predict(x_test_red, voting_models)
        pred_one_by_one = predict_feature_based(df_X_test, trained_best_models)
        prediction = (pred_voting+pred_one_by_one)/2
        
        for i in range(0, len(test_index)):
            final_prediction[test_index[i],:] = prediction[i]
        
        Five_fold_predictions.append(prediction)
        Mse_results.append(mse(prediction, y_test))
        Mae_results.append(mae(prediction, y_test))
        for i in range(y_test.shape[0]):
            print(mae(prediction[i,:],y_test[i,:]))
        #print("pearson:",pearsonr(prediction.flatten(),y_test.flatten()))
        #print("mae:",mae(prediction, y_test))
        #print("mse:",mse(prediction, y_test))
    print(sum(Mse_results)/5)
    print(np.std(Mse_results))
    print(sum(Mae_results)/5)
    print(np.std(Mae_results))

# ********** MAIN PROGRAM ********** #

train_X,train_y, test_X = load_data("train_t0.csv","train_t1.csv", "test_t0.csv")
x_train_red, x_test_red  =  preprocessing(train_X, train_y, test_X)


with open("Selected_models.pickle", "rb") as input_file:
    load_dict = pickle.load(input_file)
selected_models = load_dict["selected_models"]

# Train using voting models
voting_models = train_voting(x_train_red, train_y)

# Train using selected models
trained_best_models = train_feature_based(train_X, train_y, selected_models)

# combine the two
pred_voting = voting_predict(x_test_red, voting_models)
pred_one_by_one = predict_feature_based(test_X, trained_best_models)
prediction = (pred_voting+pred_one_by_one)/2

write_output("combined",prediction)

cv5(train_X,train_y)



