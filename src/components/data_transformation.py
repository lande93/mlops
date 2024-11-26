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
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "prepro.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function returns a ColumnTransformer object that applies transformations to numerical and categorical features.
        """
        try:
            # Define which columns are numeric and which are categorical
            numeric_features = [
        'feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5',
       'feat_6', 'feat_7', 'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12',
       'feat_13', 'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18',
       'feat_19'] # Replace with actual numeric columns
            #categorical_features = ['state', 'international plan', 'voice mail plan']
            
            
             # Replace with actual categorical columns
            

            # Create pipelines for both numerical and categorical data transformations

            # Pipeline for numeric features
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
                ('scaler', StandardScaler())  # Scale features
            ])

            # # Pipeline for categorical features
            # #categorical_transformer = Pipeline(steps=[
            # #     ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
            # #     ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical variables
            # # ])
            #                 ("onehot", OneHotEncoder(handle_unknown='ignore')),
            #     ('scaler', StandardScaler(with_mean=False))
            #     ])


       

            # Column transformer that applies the appropriate transformations to each column type
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                    #('cat', categorical_transformer, categorical_features),
                     
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

            preprocessor_obj = self.get_data_transformer_object()

            # Assuming the target column is 'target'
            target_column = ['y']
            not_imp_variables=['index']
            input_feature_train_df = train_df.drop(columns=target_column + not_imp_variables,axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=target_column + not_imp_variables,axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Splitting data into input features and target variable")

            # Get the preprocessor object
            # preprocessor_obj = self.get_data_transformer_object()
            

            # Fit and transform the training data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            print("Input Feature Train Array Shape:", input_feature_train_arr.shape)
            print("Target Feature Train Array Shape:", np.array(target_feature_train_df).shape)
            print("Input Feature Test Array Shape:", input_feature_test_arr.shape)
            print("Target Feature Test Array Shape:", np.array(target_feature_test_df).shape)


            # target_feature_train_df = target_feature_train_df.values.reshape(-1, 1)
            # target_feature_test_df = target_feature_test_df.values.reshape(-1, 1)

            

            logging.info("Applied data transformations to the train and test datasets")

            train_array = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # train_array = np.c_[input_feature_train_arr, target_feature_train_df]
            # test_array = np.c_[input_feature_test_arr, target_feature_test_df]

      
        
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        

        except Exception as e:
            logging.error(f"Error in data transformation process: {e}")
            raise CustomException(e, sys)


