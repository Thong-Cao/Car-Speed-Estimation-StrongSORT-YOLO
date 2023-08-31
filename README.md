# Credit Card Fraud Detection System

## Table of Contents
- [About the Dataset](#about-the-dataset)
- [Source of Simulation](#source-of-simulation)
- [Project Description](#project-description)
- [Features](#features)
- [Algorithms Used](#algorithms-used)
- [Usage](#usage)
- [Results](#results)


## About the Dataset
This repository contains a simulated credit card transaction dataset containing both legitimate and fraudulent transactions. The dataset covers transactions that occurred from January 1st, 2019 to December 31st, 2020. It encompasses credit card transactions involving 1000 customers interacting with a pool of 800 merchants.

## Source of Simulation
The dataset was generated using the **Sparkov Data Generation** tool developed by Brandon Harris, available on [GitHub](https://github.com/username/sparkov-data-generation). The simulation was conducted for the period from January 1st, 2019 to December 31st, 2020. The generated files were aggregated and transformed into a standardized format for ease of use.

## Project Description
The objective of this project is to build a credit card fraud detection system that can accurately classify customers based on their potential involvement in credit card fraud. To achieve this, a variety of powerful classification algorithms have been employed, including:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

In addition to utilizing these algorithms, feature selection using the Recursive Feature Elimination (RFE) method and data preprocessing techniques such as Synthetic Minority Over-sampling Technique (SMOTE) for addressing imbalanced datasets have been applied. The project also involves the comparison and evaluation of model results, which are saved in Excel format for further analysis.

## Features
- Legitimate and fraud credit card transactions dataset
- Various classification algorithms for fraud detection
- Feature selection using RFE
- SMOTE technique for handling imbalanced datasets
- Results saved in Excel format for comparison and evaluation

## Algorithms Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

## Results
The results of the classification models are saved in an Excel spreadsheet named `model_results.xlsx`. This file includes detailed information about each algorithm's performance metrics, such as accuracy, precision, recall, and F1-score, allowing for easy comparison and evaluation of the models.

