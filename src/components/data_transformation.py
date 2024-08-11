import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler 

from src.logger import logging
from src.exception import Custom_Exception
import os 
from src.utils import save_object

@dataclass
class Data_Transformation_config:
    data_transformation=os.path.join('artifact','preprocessor.pkl')
    
class Data_Transformation:
    def __init__(self):
        self.transformation=Data_Transformation_config()
        
    def get_data_transformed(self):
        logging.info('Making data transformation pickle file')
        
        try:
            logging.info('Making numerical and categorical columns')
            numerical_cols=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
            categorical_cols=['workclass','education','marital-status','occupation','relationship','race','sex','country']
            
            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]
            )
            
            numerical_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', MinMaxScaler())
                ]
            )
            
            logging.info('Numerical and categorical pipelines have been made')
            
            preprocessor=ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, numerical_cols),
                    ('cat_pipeline', categorical_pipeline, categorical_cols)
                ]
            )
            
            logging.info('Transformation has been completed')
            return preprocessor
            
        except Exception as e:
            raise Custom_Exception(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            target_col = 'salary'
            logging.info('Initiating the data transformation')
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Train and test CSVs have been read successfully')
            
            preprocessor_obj = self.get_data_transformed()
            
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            logging.info('Train split has been completed')
            
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            logging.info('Test data has been split')
            
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)
            logging.info('Preprocessor has been applied to train and test independent features')
            
            # Apply RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = ros.fit_resample(X_train_transformed, y_train)
            logging.info('Applied RandomOverSampler to the training data')
            
            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_transformed, y_test]
            
            logging.info('Data transformation has been completed')
            
            save_object(file_path=self.transformation.data_transformation, obj=preprocessor_obj)
            logging.info('Preprocessor pickle has been saved')
            
            return train_arr, test_arr, self.transformation.data_transformation
            
        except Exception as e:
            raise Custom_Exception(e, sys)
