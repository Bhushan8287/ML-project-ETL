import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:    
            model_path='artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Number_of_Customers_Per_Day:int, Average_Order_Value:float, Marketing_Spend_Per_Day:float):
        self.Number_of_Customers_Per_Day = Number_of_Customers_Per_Day
        self.Average_Order_Value = Average_Order_Value
        self.Marketing_Spend_Per_Day = Marketing_Spend_Per_Day
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input = {
                "Number_of_Customers_Per_Day": [self.Number_of_Customers_Per_Day],
                "Average_Order_Value": [self.Average_Order_Value],
                "Marketing_Spend_Per_Day": [self.Marketing_Spend_Per_Day]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)  