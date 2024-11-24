import os
import sys
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from dataclasses import dataclass

import joblib

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the ModelTrainer. Defines the path where the best model will be saved.
    """
    trained_model_file_path = os.path.join("artifacts", "best_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # Define models to train and evaluate
            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                #"AdaBoost": AdaBoostClassifier(),
                "SVM": SVC(),
                "Logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier()
                #"DecisionTree": DecisionTreeClassifier(),
                #"CatBoostClassifier": CatBoostClassifier(verbose=False)
            }
            params = {
                        "Random Forest": {
                            'n_estimators': [10, 50, 100],
                            'max_depth': [None, 10, 20]
                    },
                        "SVM": {
                            'C': [0.1, 1, 10],
                            'kernel': ['linear', 'rbf']
                    },
                        "Gradient Boosting": {
                            'learning_rate': [0.1, 0.01],
                            'n_estimators': [50, 100]
                    },
                        "Logistic Regression": {
                            'C': [0.1, 1, 10],
                            'solver': ['liblinear', 'saga']
                    },
                        "KNN": {
                            'n_neighbors': [3, 5, 10],
                            'weights': ['uniform', 'distance']
                    }
                    }


            # Evaluate all the models and get their accuracy scores
            logging.info("Evaluating models...")
            model_report:dict =evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=params)

            # Extracting the best model based on the highest accuracy
            best_model_score = max(sorted(model_report.values()))
            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found with an accuracy score greater than 0.6", sys)

            # # Saving the best model
            # best_model = models[best_model_name]
            #save_object(self.model_trainer_config.trained_model_file_path, best_model)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model saved: {best_model_name} with accuracy: {best_model_score}")

            # return best_model_score, best_model_name
            predicted=best_model.predict(X_test)

            best_score = accuracy_score(y_test, predicted)
            return best_score

        except Exception as e:
            logging.error(f"Error occurred in model training: {e}")
            raise CustomException(e, sys)


        

        
    # Data ingestion


