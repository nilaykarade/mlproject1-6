from src.exception import CustomException
import sys
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def model_prediction(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_obj_path='artifacts\preprocessor.pkl'
            model=load_object(model_path)
            preprocessor_obj=load_object(preprocessor_obj_path)
            data_scaled=preprocessor_obj.transform(features)
            predictions=model.predict(data_scaled)
            return predictions


        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                km:float,
                oprice:float,
                year:int,
                fuel_type:str,
                transmission:str,
                ) :
        self.km=km
        self.oprice=oprice
        self.year=year
        self.fuel_type=fuel_type
        self.transmission=transmission

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "Kms_Driven":[self.km],
                "Present_Price":[self.oprice],
                "Year":[self.year],
                "Fuel_Type":[self.fuel_type],
                "Transmission":[self.transmission]
            }
            
            df=pd.DataFrame(custom_data_input_dict)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        