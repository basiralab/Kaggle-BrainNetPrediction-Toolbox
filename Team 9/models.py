'''
Istanbul Technical University - Learning From Data course, 2020-2021 Fall term project.

Team:
Mehmet Arif Demirtaş - demirtasm18@itu.edu.tr
Mehmet Fatih Teloğlu - teloglu17@itu.edu.tr
Habib Bartu Gökalp - gokalph16@itu.edu.tr
Zeynep Gürler - gurler17@itu.edu.tr

This file includes the estimator classes and functions for our proposed prediction model.
'''
import numpy as np
from sklearn.base import clone, BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
np.random.seed(1)


def feature_selection(X_train, y_train, n_selected_features):
    dif = np.abs(X_train - y_train)
    dif_sum = np.sum(dif, axis=0)
    indices_high = np.argsort(dif_sum)[-n_selected_features:]
    indices_low = np.argsort(dif_sum)[:-n_selected_features]

    X_train_low = np.copy(X_train)
    X_train_high = np.copy(X_train)
    y_train_low = np.copy(y_train)
    y_train_high = np.copy(y_train)
    X_train_low[:, indices_low] = 0
    y_train_low[:, indices_low] = 0
    X_train_high[:, indices_high] = 0
    y_train_high[:, indices_high] = 0

    regressor_high = RandomForestRegressor()
    regressor_low = RandomForestRegressor()

    regressor_high.fit(X_train_high, y_train_high)
    regressor_low.fit(X_train_low, y_train_low)

    return regressor_high, regressor_low, indices_high, indices_low

def predict_(x, indices_high, indices_low, reg_high, reg_low):
    x_high = x.copy()
    x_low = x.copy()
    x_high[:, indices_high] = 0
    x_low[:, indices_low] = 0

    pred_high = reg_high.predict(x_high)
    pred_low = reg_low. predict(x_low)

    return pred_high + pred_low

class BallTreeWeighted(BaseEstimator, RegressorMixin, TransformerMixin):
    '''Nearest neighbor approximator using ball-tree distance
    This model takes in the training data and finds an embedding for
    the space that maps all features to a lower dimensional space that
    includes n_components, finds embeddings for test subjects in this space,
    finds the nearest k neighbors to test sample, computes a weighted average
    based on distances to predict test sample at next timepoint.
    '''

    def __init__(self, n_components=55, k=25):
        self.embedder = TruncatedSVD(n_components=n_components, n_iter=20, random_state=42)
        self.n_components = n_components
        self.embeddings = None
        self.outputs = None
        self.k = k

    def fit(self, X, y):
        self.embedder.fit(X)
        self.embeddings = self.embedder.transform(X)
        self.outputs = y
        return self

    def predict(self, X):
        predictions = []
        test_embeddings = self.embedder.transform(X)
        for i in range(X.shape[0]):
            embedding = np.reshape(test_embeddings[i], (1, self.n_components))
            total = np.append(self.embeddings, embedding, axis=0)
            neighbors = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(total)
            distances, indices = neighbors.kneighbors(total)

            sum = np.zeros((1, 595), dtype=float)
            weight_sum = 0

            for j in range(1, self.k + 1):
                #                max_val, min_val = 1/np.amin(distances[-1, 1:]), 1/np.amax(distances[-1, 1:])
                #                weight = (1/distances[-1, j] - min_val) / (max_val - min_val)
                weight = 1 / distances[-1, j]
                sum += self.outputs[indices[-1, j], :] * weight
                weight_sum += weight
            average = sum / weight_sum
            predictions.append(average[0])
        return np.stack(predictions)


class FeatureClusterRF(BaseEstimator, RegressorMixin, TransformerMixin):
    '''Random Forest Regressor with clustering and feature selection
    This model uses Random Forest regressors and KMeans clustering algorithm
    in an ensemble learning manner, by generating a Random Forest
    regressor for each cluster in the data as predicted by unsupervised KMeans.
    KMeans uses the residuals between timepoints (X-y), so to generate
    residuals for test set, an initial Random Forest regressor is trained on
    all training data to create intermediate predictions.
    A feature selection routine is performed to generate regressors for
    important (high) and non-important (low) features in the data seperately, to generate
    predictions more robust to outliers.
    Args:
        k (int): Number of clusters
        n_selected_features (int): Number of features to be selected important
    '''

    def __init__(self, n_clusters=5, n_selected_features=50):
        self.init_regressor = RandomForestRegressor()
        self.regressors = [RandomForestRegressor()
                           for _ in range(n_clusters)]

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        self.n_clusters = n_clusters
        self.n_selected_features = n_selected_features

    def fit(self, X, y):
        self.regressor0_high, self.regressor0_low, self.index0_high, self.index0_low = feature_selection(X, y, self.n_selected_features)
        y_pred = predict_(X, self.index0_high, self.index0_low, self.regressor0_high, self.regressor0_low)
        self.kmeans.fit(np.abs(X - y_pred))

        self.regressors_high = []
        self.regressors_low = []
        self.index_high = []
        self.index_low = []
        for n in range(self.n_clusters):
            regressor_high, regressor_low, indices_high, indices_low = feature_selection(
                X[self.kmeans.labels_[:X.shape[0]] == n], y[self.kmeans.labels_[:X.shape[0]] == n],
                self.n_selected_features)
            self.regressors_high.append(regressor_high)
            self.regressors_low.append(regressor_low)
            self.index_high.append(indices_high)
            self.index_low.append(indices_low)
        return self

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            reshaped = np.reshape(X[i], (1, 595))
            label = self.kmeans.predict(np.abs(X[i] - predict_(np.array([X[i]]), self.index0_high, self.index0_low, self.regressor0_high, self.regressor0_low)))

            self.test_high = X[i].copy()
            self.test_low = X[i].copy()
            self.test_low[self.index_low[label[0]]] = 0
            self.test_high[self.index_high[label[0]]] = 0
            y_predhigh = self.regressors_high[label[0]].predict(np.array([self.test_high]))
            y_predlow = self.regressors_low[label[0]].predict(np.array([self.test_low]))
            y_pred = y_predhigh + y_predlow
            preds.append(y_pred[0])
        return np.stack(preds)


class MeanEnsembler(BaseEstimator, RegressorMixin, TransformerMixin):
    '''Given a list of models, fits them on training data and aggregates
    their predictions by mean if no weights are provided or by weighted
    average if a weights list is given.'''

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else np.ones((len(models)))

    def fit(self, X, y):
        self.models_ = [clone(model) for model in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.stack([model.predict(X) for model in self.models_], axis=2)
        return np.average(predictions, axis=2, weights=self.weights)
