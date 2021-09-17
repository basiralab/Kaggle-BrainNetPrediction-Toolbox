"""
This code uses built-in cross validation to find the most optimal hyper parameters for each model that is used in the mapping function.
It is recommended but not required to run this code before running main code to get a stronger model. Please keep in mind that running
this code uses all the cpu cores available, and will probably take a lot of time
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

def create_pipeline(regressor):
    """
    Parameters:
    regressor: Regressor object instance to be optimized.
    ---------------------------------------------------------------------
    Creates a pipeline object that can be used in RandomizedGridSearchCV.
    Pipelines consist of a feature selection algorithm SelectKBest that
    reduces the number of features to 20, and a regressor.
    """
    return Pipeline([
        ("selector", SelectKBest(mutual_info_regression, k=350)),
        ("regressor", regressor)
        ])

def grad_params(data, label):
    """
    Parameters:
    data: The training data that will be used for classification.
    label: The dataset that contains all the labels.
    ---------------------------------------------------------------
    This function will iterate over each of 595 labels and performs
    3-fold cross-validation to split the training data into test
    and train groups. Then it trains the given base model (Gradient
    boosting regressor in this case) 50 times and compares the
    results. Then it chooses the best results and saves it in the
    DataFrame. After performing this for each label, the function
    writes the parameters into a csv file.
    """

    param_dist = {
    "regressor__n_estimators": [5, 10, 25, 50],            #number of base models to be trained
    "regressor__learning_rate": [0.01, 0.1, 0.5, 0.7, 1],  #multiplies the contribution of each tree with this
    "regressor__loss": ["ls", "lad", "huber", "quantile"], #loss function to minimize
    "regressor__alpha": [0.1, 0.5, 0.9],                   #alpha value for huber and quantile loss functions
    "regressor__max_depth": [2, 3]                         #max depth of the base DecisionTreeRegressor
    }
    #initialize dataframe to store
    param = pd.DataFrame(index=["n_estimators", "max_depth", "loss", "learning_rate", "alpha"])
    print("Start optimizing GradientBoostingRegressor")
    for col in label:   #iterates over the labels while training a model each time
        print(col)
        #creates pipeline
        pipeline = create_pipeline(GradientBoostingRegressor())
        #built-in cross-validation searcher
        params = RandomizedSearchCV(pipeline,
        param_distributions = param_dist,
        cv=3,
        n_iter=25,
        n_jobs=-1,
        refit = False,
        scoring = "neg_mean_squared_error"
        )
        params.fit(data, label[col])
        #stores best parameters
        param[col] = params.best_params_.values()
    #writes found parameters to file
    param.to_csv("grad_params.csv", index_label="ID")

def knr_params(data, label):
    """
    Parameters:
    data: The training data that will be used for classification.
    label: The dataset that contains all the labels.
    ---------------------------------------------------------------
    Same as grad_params function, but this function finds optimal
    parameters for K Neighbors Regressor model.
    """
    param_dist = {
    "regressor__n_neighbors": [3,5,7,10,15,20,50], #number of neighbors to use
    "regressor__weights": ["uniform", "distance"], #weight function to use in prediction
    "regressor__algorithm": ["auto", "ball_tree", "kd_tree", "brute"], #algorithm to compute nearest neighbors
    "regressor__p": [1, 2] #1 means manhattan distance, 2 means euclidian distance
    }

    param = pd.DataFrame(index=["algorithm", "n_neighbors", "p", "weights"])
    print("Start optimizing KNeighborsRegression")
    for col in y_train:
        print(col)
        #creates pipeline
        pipeline = create_pipeline(KNeighborsRegressor())
        #built-in cross-validation searcher
        params = RandomizedSearchCV(pipeline,
        param_distributions = param_dist,
        cv = 3,
        n_iter = 25,
        n_jobs = -1,
        refit = False,
        scoring = "neg_mean_squared_error"
        )
        params.fit(data, label[col])
        #stores best parameters
        param[col] = params.best_params_.values()
    #writes found parameters to file
    param.to_csv("knr_params.csv", index_label="ID")

def ada_params(data, label):
    """
    Parameters:
    data: The training data that will be used for classification.
    label: The dataset that contains all the labels.
    ---------------------------------------------------------------
    Same as grad_params function, but this function finds optimal
    parameters for Ada Boosting Regressor model.
    """
    param_dist = {
    "regressor__n_estimators": [5, 10, 25, 50],
    "regressor__learning_rate": [0.01, 0.05, 0.1, 0.3, 0.7, 1],
    "regressor__loss": ["linear", "square", "exponential"]
    }

    param = pd.DataFrame(index=["learning_rate", "loss", "n_estimators"])
    print("Start optimizing AdaBoostingRegresion")
    for col in y_train:
        print(col)
        #creates pipeline
        pipeline = create_pipeline(AdaBoostRegressor())
        #built-in cross-validation searcher
        params = RandomizedSearchCV(pipeline,
        param_distributions = param_dist,
        cv = 3,
        n_iter = 25,
        n_jobs = -1,
        refit = False,
        scoring = "neg_mean_squared_error"
        )
        params.fit(data, label[col])
        #stores best parameters
        param[col] = params.best_params_.values()
    #writes found parameters to file
    param.to_csv("params.csv", index_label="ID")

X_train, y_train = pd.read_csv("data/train_t0.csv", index_col="ID"), pd.read_csv("data/train_t1.csv", index_col="ID")

grad_params(X_train, y_train)

knr_params(X_train, y_train)

ada_params(X_train, y_train)