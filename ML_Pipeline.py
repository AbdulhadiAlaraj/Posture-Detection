import os

from sklearn.metrics import accuracy_score, classification_report # Accuracy metrics 
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv(r'coords.csv')
labels = dataset['class'].replace({'proper': 1, 'improper': 0})
print("These are the categories: ", labels)
features = dataset.drop('class', axis=1)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define your pipelines for different models
pipelines = {
    'Logistic_Regression': make_pipeline(StandardScaler(), LogisticRegression()),
    'Ridge_Classifier': make_pipeline(StandardScaler(), RidgeClassifier()),
    'Lin_Support_Vector_Class': make_pipeline(StandardScaler(), LinearSVC()),
    'KNearest_Neighbors': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'Naive_Bayes': make_pipeline(StandardScaler(), GaussianNB())
    }

# Fit the models
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(x_train, y_train)
    fit_models[algo] = model

    model_path = f'{algo}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

for algo, model in fit_models.items():
    yhat = model.predict(x_test)
    print(algo,'\n', classification_report(y_test, yhat))