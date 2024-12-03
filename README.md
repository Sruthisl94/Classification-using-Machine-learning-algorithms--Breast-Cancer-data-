# Classification using Machine Learning Algorithms--Breast Cancer Data

## Summary:
This project focuses on classifying Breast cancer data using machine learning algorithms. The primary goal is to evaluate model performance and provide insights into the most effective techniques for this critical healthcare problem.

## Steps
## 1. Data Loading and Exploration
Dataset: Loading breast cancer data from sklearn.datasets
Features: Includes attributes like mean radius, texture, smoothness, compactness, and more.
Target: Binary classification: 0 and 1
## 2. Data Preprocessing
Missing values and cleaning of data is done using python pandas library. The dataset have no duplicates and no other inconsistencies.
The outliers are removed using IQR method.
Scaling: Used StandardScaler for normalization.
Feature Selection: SelectKBest 
3. Model Implementation: The following model algorithms are implemented on the dataset
* Logistic Regression
Linear model for binary classification.It uses the logistic (sigmoid) function to predict the probability of an instance belonging to a particular class. If the probability exceeds a threshold (usually 0.5), the instance is classified as one class; otherwise, it belongs to the other.
* Support Vector Machines (SVM)
   Supervised learning model for classification and regression. SVM finds the hyperplane that best separates data points of different classes with the maximum margin. It can also use kernels to handle non-linear decision boundaries.
* K-Nearest Neighbors (KNN)
   Instance-based learning algorithm. KNN classifies a data point based on the majority class of its k nearest neighbors in the feature space. It is highly dependent on the distance metric and the choice of 𝑘
* Random Forest Classifier
   Ensemble learning method. Random Forest builds multiple decision trees during training and combines their outputs (via averaging or majority voting) to improve accuracy and reduce overfitting.
* Decision Tree Classifier
   Tree-based model for classification and regression. It splits the dataset into subsets based on feature values, creating a tree structure where each node represents a decision based on a feature. The leaves represent the final class labels.
## Code Highlights:
Used sklearn for model implementation and evaluation.
4. Performance Evaluation
Evaluated all models using the following metrics:
Accuracy: Overall correctness of predictions.
Confusion Matrix: Breakdown of true label and predicted label.

## Results 
Best Performing Algorithm: Achieved the highest accuracy (0.925) with multiple algorithms, including Logistic Regression and SVM.
Worst Performing Algorithm: Random Forest Classifier with 0.9125 accuracy
Insights:
Simpler models like Logistic Regression and SVM perform well due to the linear separability of the dataset.
Feature selection plays a crucial role in enhancing model accuracy and interpretability.
Algorithm	Accuracy
Logistic Regression	0.925
SVM	0.925
KNN	0.925
Random Forest	0.9125
Decision Tree	0.925
## Conclusion
This project highlights the importance of preprocessing and feature selection in machine learning workflows.
Logistic Regression and SVM emerged as top-performing algorithms for this dataset.

