# Machine Learning Methods for Brain Connectivity Evolution Prediction from a Single Observation
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

The brain connectivity map dataset includes 230 subjects in total from [OASIS-2 Dataset](https://www.oasis-brains.org/) referenced below.

```
Marcus, D.S., Fotenos, A.F., Csernansky, J.G., Morris, J.C., Buckner, R.L., 2010.
Open access series of imaging studies: longitudinal mri data in nondemented and demented older adults.
Journal of cognitive neuroscience 22, 2677–2684.
```

This set consists of a longitudinal collection of 230 subjects aged 60 to 96. For each subject, T1-weighted MRI scans for 3 timepoints with interval of at least one year is included in the dataset. For the dataset used in the competition, cortical morphological networks are derived from structural T1-weighted MRIs of 2 consecutive timepoints (t0 and t1) for each subject. These connectivity maps are represented by 595 morphological connectivity features.

The train-test split for total of 230 subjects is as follows:
- **Training set:** 150 subjects
- **Public test set:** 40 subjects
- **Private test set:** 40 subjects

Teams trained their pipelines on given t0 and t1 maps of training set. They given only the t0 maps of public test set and submitted their predictions for t1 maps to the system for evaluation and examined their results. The private test set completely kept hidden during the competition and after the competition ends, the submitted pipelines are tested also on the private test set.

# Please cite the following paper when using BrainNet-Prediction-ToolBox:

```
TBA

```
# Paper link on arXiv
https://arxiv.org/abs/2109.07739
