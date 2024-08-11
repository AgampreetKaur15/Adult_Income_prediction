import sys
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from src.exception import Custom_Exception
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = '/home/googlyji/ml_projects/ml projects /adult census prediction /artifact/model.pkl'
            preprocessor_path = '/home/googlyji/ml_projects/ml projects /adult census prediction /artifact/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise Custom_Exception(e, sys)

class CustomData:
    def __init__(self,
                 age: float,
                 fnlwgt: float,
                 education_num: float,
                 capital_gain: float,
                 capital_loss: float,
                 hours_per_week: float,
                 workclass: str,
                 education: str,
                 marital_status: str,
                 occupation: str,
                 relationship: str,
                 race: str,
                 sex: str,
                 country: str):
        
        self.age = age
        self.fnlwgt = fnlwgt
        self.education_num = education_num
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.workclass = workclass
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.country = country

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'age': [self.age],
                'fnlwgt': [self.fnlwgt],
                'education-num': [self.education_num],
                'capital-gain': [self.capital_gain],
                'capital-loss': [self.capital_loss],
                'hours-per-week': [self.hours_per_week],
                'workclass': [self.workclass],
                'education': [self.education],
                'marital-status': [self.marital_status],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'race': [self.race],
                'sex': [self.sex],
                'country': [self.country]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise Custom_Exception(e, sys)

# Example usage:
# custom_data = CustomData(
#     age=39.0, fnlwgt=77516.0, education_num=13.0, capital_gain=2174.0,
#     capital_loss=0.0, hours_per_week=40.0, workclass='State-gov', education='Bachelors',
#     marital_status='Never-married', occupation='Adm-clerical', relationship='Not-in-family',
#     race='White', sex='Male', country='United-States'
# )
# df = custom_data.get_data_as_data_frame()
# prediction_pipeline = PredictionPipeline()
# predictions = prediction_pipeline.predict(df)
# print(predictions)
