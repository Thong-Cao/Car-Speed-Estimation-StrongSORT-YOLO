import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from feature_engine.selection import RecursiveFeatureElimination
from xgboost import XGBClassifier
import time
from sklearn.metrics import classification_report
from sklearn import preprocessing
pd.set_option('display.max_columns', None)
from imblearn.over_sampling import SMOTE




df = pd.read_csv("./fraudTrain.csv")


label_encoder = preprocessing.LabelEncoder()
for col in df.columns:
    df[col]= label_encoder.fit_transform(df[col])

columns_to_drop = ['Unnamed: 0', 'cc_num', 'merchant', 'first', 'last', 'zip', 'trans_num', 'trans_date_trans_time', 'unix_time']

df = df.drop(columns=columns_to_drop)

############################################################################################ PHASE1

def evaluateModel(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    report = classification_report(y_test, y_test_pred, output_dict=True)
    accuracy = report['accuracy']
    precision = [report[str(i)]['precision'] for i in range(2)]
    recall = [report[str(i)]['recall'] for i in range(2)]
    f1_score = [report[str(i)]['f1-score'] for i in range(2)]
    return accuracy, precision, recall, f1_score, classification_report(y_test, y_test_pred)


# Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "Support Vector Machine (RBF Kernel)": SVC(),
    "Neural Network": MLPClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBOOST": XGBClassifier()
}

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)


sm = SMOTE()
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())


# Initialize an empty dictionary to store the results
results = {}

# Iterate over each model
for name, model in models.items():
    start_time = time.time()
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train_res)
    X_test_scaled = sc.transform(X_test)
    print('Start training' + name)
    model.fit(X_train_scaled, y_train_res) 
    
    
    accuracy, precision, recall, f1_score, report = evaluateModel(model, X_test_scaled, y_test) 
    results[name] = {
            'Accuracy': accuracy,
            'Precision (Class 0)': precision[0],
            'Precision (Class 1)': precision[1],
            'Recall (Class 0)': recall[0],
            'Recall (Class 1)': recall[1],
            'F1-Score (Class 0)': f1_score[0],
            'F1-Score (Class 1)': f1_score[1],
            'Time_Train': time.time() - start_time
        }
    print(report)
        
           


# Create a DataFrame to display the results
results_df = pd.DataFrame(results)
results_df.index.name = 'Model'

results_df = results_df.round(3)
results_df.to_excel("./result_Smote_OnlyTrain.xlsx")
