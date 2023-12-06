from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics,preprocessing
#from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from collections import Counter
from scipy.stats import norm
import seaborn as sns; sns.set(font_scale=1.2)
import time
#import xgboost as xgb
#import lightgbm as lgb

warnings.simplefilter("ignore")
import time

def process_chunk(args):
    chunk, knn_model, y_test_chunk = args
    y_pred_knn = knn_model.predict(chunk)
    return accuracy_score(y_test_chunk, y_pred_knn)

if __name__ == "__main__":
    st = time.time()

    # Reading the data
    dfall_read_seq = pd.read_json('dfallwesad560.json', orient='split', compression='infer')
    dfall_read = dfall_read_seq.sample(frac=1, random_state=42)
    dfall = dfall_read.copy()
    #dfall
    def convert_slither(df):
        return 1 if df['label'] == 2 else 0
    dfall['label_binary'] = dfall.apply(convert_slither, axis=1)
    dfall=dfall.drop(columns=['label'])
    dfall
    Xc = dfall.iloc[:, 0:-1].values
    y = dfall.iloc[:, -1].values
    X = scale(Xc)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))
## Need to Chagne the number of core
    num_cores = 16
    chunks = np.array_split(X_test, num_cores)
    y_test_chunks = np.array_split(y_test, num_cores)
## Classifier
## NEED TO CHANGE HERE "svm.SVC(C=10, kernel='rbf',degree=3,gamma=0.1)"
    model =svm.SVC(C=10, kernel='rbf',degree=3,gamma=0.1)
    
    model.fit(X_train, y_train)

    args = [(chunk, model, y_test_chunk) for chunk, y_test_chunk in zip(chunks, y_test_chunks)]

    with Pool(processes=num_cores) as pool:
        accuracies = pool.map(process_chunk, args)

    # Calculate accuracy on the entire test set
    acc_knn = np.mean(accuracies)
    print("Mean Accuracy: ", acc_knn)

    # Calculate ROC curve and AUC on the entire test set
    y_pred_knn = model.predict(X_test)
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn, pos_label=2)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn, pos_label=2)
    auc_score = auc(fpr, tpr)
    print("AUC Score: %0.3f" % auc_score)

    # Calculate confusion matrix on the entire test set
    final_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
    print("Confusion Matrix:")
    print(final_confusion_matrix)
    print("Total time:", (time.time() - st))
