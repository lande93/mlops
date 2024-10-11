import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        
        try:
            # Step 1: Load the dataset
            logging.info(f"Reading dataset from the data")
            df = pd.read_csv("notebook\data\Data Science.csv")
            logging.info("Dataset loaded successfully")
            
            # Step 2: Create artifact directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info(f"Saving raw dataset to {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Step 3: Train-test split
            logging.info("Performing train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            logging.info(f"Saving train data to {self.ingestion_config.train_data_path}")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info(f"Saving test data to {self.ingestion_config.test_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion process completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(f"File not found: {e}", sys)
        
        except pd.errors.EmptyDataError as e:
            logging.error(f"Data file is empty: {e}")
            raise CustomException(f"Data file is empty: {e}", sys)

        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)

# Usage example
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
