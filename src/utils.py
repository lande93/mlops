import os
import sys
import dill  # dill is used for serializing objects
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle
import joblib
import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Save Python objects to a file using dill.
    
    :param file_path: str, path where the object should be saved.
    :param obj: Python object to be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open the file and dump the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        print(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)
            

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

