import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
import joblib


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function returns a ColumnTransformer object that applies transformations to numerical and categorical features.
        """
        try:
            # Define which columns are numeric and which are categorical
            numeric_features = ['account length',
 'area code',
 'number vmail messages',
 'total day minutes',
 'total day calls',
 'total day charge',
 'total eve minutes',
 'total eve calls',
 'total eve charge',
 'total night minutes',
 'total night calls',
 'total night charge',
 'total intl minutes',
 'total intl calls',
 'total intl charge',
 'customer service calls'] # Replace with actual numeric columns
            categorical_features = ['state', 'phone number', 'international plan', 'voice mail plan'] # Replace with actual categorical columns

            # Create pipelines for both numerical and categorical data transformations

            # Pipeline for numeric features
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
                ('scaler', StandardScaler())  # Scale features
            ])

            # Pipeline for categorical features
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
                ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical variables
            ])

            # Column transformer that applies the appropriate transformations to each column type
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            logging.info("Data transformer object created successfully")
            return preprocessor

        except Exception as e:
            logging.error(f"Error in creating data transformer object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function applies the transformations to the train and test datasets.
        """
        try:
            # Load training and test data
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data")

            # Assuming the target column is 'target'
            target_column = "churn"
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info("Splitting data into input features and target variable")

            # Get the preprocessor object
            preprocessor = self.get_data_transformer_object()

            # Fit and transform the training data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applied data transformations to the train and test datasets")

            # Save the preprocessor object for later use
            joblib.dump(preprocessor, self.transformation_config.preprocessor_obj_file_path)
            logging.info(f"Preprocessor saved at {self.transformation_config.preprocessor_obj_file_path}")

            return (
                input_feature_train_arr, target_feature_train_df,
                input_feature_test_arr, target_feature_test_df,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation process: {e}")
            raise CustomException(e, sys)

# Usage example:
if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    data_transformation.initiate_data_transformation(train_path, test_path)
