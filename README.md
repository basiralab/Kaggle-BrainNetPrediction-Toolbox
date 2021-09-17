# Machine Learning Methods for Brain Connectivity Estimation from Single Observation
==============================================================

Thw BrainNet Prediction Toolbox is a compilation of the proposed ML pipelines for brain connectivity map estimation.
Pipelines consisting of data pre-processing, dimensionality reduction and regression models are created with help of key ML libraries such as scikit-learn <https://scikit-learn.org/stable/index.html>, and xgboost <https://xgboost.ai/>.

# Introduction

The estimation of brain connectivity map in a future timestep makes it possible to diagnose the connectivity related diseases in earlier phases or detect the potential anomalies
through brain evolution. Given that the predictive ability of artificial learning algorithms have been improving gradually in recent years, the prediction of brain connectivity
task also became of concern for machine learning researchers. However the task is relatively less explored and there are a few works focused on machine learning solutions for
brain evolution prediction.

In order to cover this gap in the literature, an in-class **Kaggle challenge** is organized where we encouraged the participants to experiment with various machine learning
methods. Participants compete within teams where they developed their own pipelines to estimate the connectivity map in a further timestep given the connectivity map at an initial
timestep. **The pipelines of the best performing 20 teams are included in this repo which involve a large variety of built-in machine learning methods for data pre-processing, dimensionality
reduction and regression.**

![BrainNet](Fig1.png)

The challenge results are evaluated in 3 different experimental setups and the teams are ranked based on their MAE and PCC scores:
* **Public scores:** Scores on the first half of the test set. Teams were able to examine their public scores and rank during the competition.
* **Private scores:** Scores on the hidden part of the test set. Teams were unable to see their results on this part before competition ends.
* **5F CV scores:** Scores on the 5F CV on given training set. 

Teams are first ranked based on their MAE and PCC scores separately for each setup having 6 different rankings. Then the overall ranks are determined based on the average of these
ranks. So that the results better reveal the generalizability of the proposed pipelines on both training and test sets. Team ranks and employed methods are given at table below.

![BrainNet](Fig2.png)

The scores and rankings of top-3 teams are given below.

![BrainNet](Fig3.png)

# Installation

The source codes have been tested with Python 3.7. There is no need of GPU to run the codes.

Required Python Modules:

* csv
* numpy
* pandas
* scipy
* xgboost
* warnings
* matplotlib
* scikit-learn

# Dataset format

The brain connectivity map dataset includes 230 samples in total as 150 samples are given as training set and 80 samples were in the test set. Brain connectivity maps
of individuals are represented by 595 morphological connectivity features. Source codes load the data from CSV files via load_data function, if you intend to
run codes on another dataset you may need to make necessary modifications on the related function. 

# Please cite the following paper when using BrainNet-Prediction-ToolBox:

```
TBA

```
Paper link on arXiv:
TBA
