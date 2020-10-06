### Project motivation
Online transaction fraud is becoming every industry's problem and it leads to a significant loss to the online sellers. Even though fraud detection is every company's interest, the data are usually not available due to privacy. An e-commerce payment solution company, Vesta, provided a large dataset of 1 million transactions with masked features on Kaggle competition in 2019. The dataset has about 433 features with vast majority of numeric variables that were engineered by the company. The dataset has many relevant factors that are correlated with fraud, so it seemed to be worthwhile to try finding the optimal model for the problem. The final model was XGBoost which outperformed random forest, extra trees, and logistic regression model. The predicted probabilities were calibrated using Isotonic Regression as they were over-forecast due to class imbalance. The important features based on permutation importance with scoring as recall were mostly count data, time-delta, and transaction amount.

### Directory
Note: files that are over 100MB are not uploaded, thus 'data' folder does not exist in this repository.
There is one data folder, helper functions, Variable Description text file, and 6 notebooks with the relevant folders in this project:<br>
- 'data' folder contains train/val/test sets and imputed train/val/test data matrices. Also, there are train/val sets imputed with Bayesian Ridge regression which was not used for this project. There is 'imputation_train+val' folder that contains imputed train+val dataset and imputed test set which were also not used for this project due to the framework of calibrators and XGBoost.<br>
- 'helper_functions.ipynb' has the custom functions such as creating dummy variables, plotting, and printing evaluations of the model that are used throughout modeling.<br>
- 'Variable Description.txt' is the aggregation of the discussions about what variables are written by the host of the Kaggle competition and the participant users.
1. EDA: Exploratory Analysis. Relevant files are in the folder 'EDA'
2. clean_data: Converts/imputes/splits the raw dataset
3. logistic_regression: A notebook that implements logistic regression model. 'logistic_regression' folder has the final model and calibrators
4. random_forest: Implements random forest model and tunes the hyperparameters. The final model and lists of F scores for each hyperparameter is saved in 'random_forest' folder.
5. extra_trees: Implemented the same way as 'random_forest.ipynb' and its folder as well.
6. xgboost: Implements two candidate models and there are 4 relevant folders inside 'xgboost' folder
    - models: contains two models one with .20 column sample rate and the other with .05 column sample rate
    - hyperparameter_tuning: fbeta_scores_boot(.1) is a dictionary of F score list of 10 bootstrap samples each of which has 10% of the training set for computational purpose. It looks like `{bootstrap sample1: [Fscore1, ..., Fscore10], ..., bootstrap sample10: [Fscore1, ..., Fscore10]}`. max_beta_colrates_boot is a list of 10 colrates that produced the maximum F score for each bootstrap sample. Thus, colrate for the final model was chosen based on the majority of colrates from this list.
    - feature_importance: contains one permutation_importance data frame in pickle and csv format and one feature_importances object for the final model.
    - 'props.pkl' contains proportions of positives for each predicted probability to help us estimate the calibration of the probabilities

    
### Description of Data Analysis
The models were trained based on F_{1.5} score where recall is 1.5 times more important than precision since the goal of this project is to catch transactional fraud rather than reducing the hassles with innocent customers/legal transactions. The feature importances/permutation importances help us understand which type of features that the models preferred. Confusion matrix shows the performance of the model on the test set.

### Design Decisions
The data has 383 numeric features and 50 categorical features and has a lot of missing values. 49% of the features are missing 50% or more values. For all the models, the missing values were imputed with median for numeric features and mode for categorical features. XGBoost does handle missing data, but it didn't seem to perform better than imputed data. Among 50 categorical variables, some had such high cardinality like 500 or 15,500 which seemed impossible to create all the dummies for. Thus, categorical variables were selected manually by the user based on the cardinality and the correlation with the target('isFraud') and those variables were created with dummies.
The numeric variables were all included in the model to incorporate all the information available since the correlation between a target and a feature was at most 0.38.
PCA was actually considered and experimented in the beginning of this project, but PCA flattens the feature dimensions that loses the meaning of classification in a similar way that happens in cluster analysis as well.
Bayesian Ridge Regression as an iterative imputer was considered in the beginning, but somehow it performed poorly with the models - they were ran about 11 hours on GCP with n1-standard-16(16vCPU 60GB memory). 

### Resources
#### Dataset
- IEEE-CIS Fraud Detection, *Can you detect fraud from customer transactions?*, IEEE Computational Intelligence Society, 2019, URL: https://www.kaggle.com/c/ieee-fraud-detection
- Kaggle discussion about variables, URL: https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203

#### Python Packages
##### pandas
Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)
##### numpy
* Travis E. Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
* Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux. The NumPy Array: A Structure for Efficient Numerical Computation, Computing in Science & Engineering, 13, 22-30 (2011), DOI:10.1109/MCSE.2011.37

##### sklearn
Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay. Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, 12, 2825-2830 (2011)
##### matplotlib
John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55
##### seaborn
Waskom, M., Botvinnik, Olga, O&#39;Kane, Drew, Hobson, Paul, Lukauskas, Saulius, Gemperline, David C, … Qalieh, Adel. (2017). mwaskom/seaborn: v0.8.1 (September 2017). Zenodo. https://doi.org/10.5281/zenodo.883859
##### xgboost
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939785
##### eli5
© Copyright 2016-2017, Mikhail Korobov, Konstantin Lopuhin Revision 2497ec37. URL: https://eli5.readthedocs.io/
##### prince
License: MIT License (MIT)
Author: Max Halford
URL: https://pypi.org/project/prince/
##### ipython
Fernando Pérez and Brian E. Granger. IPython: A System for Interactive Scientific Computing, Computing in Science & Engineering, 9, 21-29 (2007), DOI:10.1109/MCSE.2007.53
##### LightGBM
URL: https://lightgbm.readthedocs.io/en/latest/index.html

#### References
- Documentation of XGBoost, URL: https://xgboost.readthedocs.io/en/latest/parameter.html
- Permutation Importance, Dan Becker, Kaggle, URL: https://www.kaggle.com/dansbecker/permutation-importance
- Online payment fraud, RAVELIN INSIGHTS, URL: https://www.ravelin.com/insights/online-payment-fraud
- Feature importance — what’s in a name?, Sven Stringer, Jul 23, 2018, Medium, URL: https://medium.com/bigdatarepublic/feature-importance-whats-in-a-name-79532e59eea3
- How to Calculate Feature Importance With Python by Jason Brownlee on March 30, 2020 in Data Preparation, URL: https://machinelearningmastery.com/calculate-feature-importance-with-python/
- Entropy: How Decision Trees Make Decisions, *The simple logic and math behind a very effective machine learning algorithm*, Sam T, Jan 10, 2019 https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8
- How and When to Use a Calibrated Classification Model with scikit-learn by Jason Brownlee on September 3, 2018 in Probability, URL: https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/
- 1.15. Isotonic regression, Scikit-Learn Documentation, URL: https://scikit-learn.org/stable/modules/isotonic.html
- Title page image from https://unsplash.com/