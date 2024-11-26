import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","best_model.pkl")
            preprocessor_path=os.path.join('artifacts','prepro.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        import pandas as pd
import sys

class CustomException(Exception):
    def __init__(self, message, source):
        super().__init__(message)
        self.source = source

class CustomData:
    def __init__(
        self,
        feat_0: str,
        feat_1: str,
        feat_2: str,
        feat_3: str,
        feat_4: str,
        feat_5: int,
        feat_6: int,
        feat_7: int,
        feat_8: int,
        feat_9: int,
        feat_10: int,
        feat_11: int,
        feat_12: int,
        feat_13: int,
        feat_14: int,
        feat_15: int,
        feat_16: int,
        feat_17: int,
        feat_18: int,
        feat_19: int,
    ):
        self.feat_0 = feat_0
        self.feat_1 = feat_1
        self.feat_2 = feat_2
        self.feat_3 = feat_3
        self.feat_4 = feat_4
        self.feat_5 = feat_5
        self.feat_6 = feat_6
        self.feat_7 = feat_7
        self.feat_8 = feat_8
        self.feat_9 = feat_9
        self.feat_10 = feat_10
        self.feat_11 = feat_11
        self.feat_12 = feat_12
        self.feat_13 = feat_13
        self.feat_14 = feat_14
        self.feat_15 = feat_15
        self.feat_16 = feat_16
        self.feat_17 = feat_17
        self.feat_18 = feat_18
        self.feat_19 = feat_19

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "feat_0": [self.feat_0],
                "feat_1": [self.feat_1],
                "feat_2": [self.feat_2],
                "feat_3": [self.feat_3],
                "feat_4": [self.feat_4],
                "feat_5": [self.feat_5],
                "feat_6": [self.feat_6],
                "feat_7": [self.feat_7],
                "feat_8": [self.feat_8],
                "feat_9": [self.feat_9],
                "feat_10": [self.feat_10],
                "feat_11": [self.feat_11],
                "feat_12": [self.feat_12],
                "feat_13": [self.feat_13],
                "feat_14": [self.feat_14],
                "feat_15": [self.feat_15],
                "feat_16": [self.feat_16],
                "feat_17": [self.feat_17],
                "feat_18": [self.feat_18],
                "feat_19": [self.feat_19],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

