import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        logging.info("Entered the data transormation component")
        try:
            numerical_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']#list(df.select_dtypes(include='object').columns)
            categorical_columns = [ 'Fuel_Type', 'Seller_Type', 'Transmission']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Selling_Price"
            #numerical_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']

            #X_train
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            
            #y_train
            target_feature_train_df=train_df[target_column_name]
            
            #X_test
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            
            #y_test
            target_feature_test_df=test_df[target_column_name]

            print("input_feature_train_df------",input_feature_train_df.columns)
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            logging.info(str(input_feature_train_df.columns))
            logging.info(str(input_feature_test_df.columns))

            #apply transformation on X_train and X_test
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr                
            )
        except Exception as e:
            raise CustomException(e,sys)