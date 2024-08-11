import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object, evaluate_model
import numpy as np 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> Dict[str, float]:
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Apply SMOTE to handle class imbalance
            ros = RandomOverSampler()
            X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: Dict[str, float] = evaluate_model(
                X_train=X_train_res,
                y_train=y_train_res,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]


            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            conf_matrix = confusion_matrix(y_test, predicted)
            class_report = classification_report(y_test, predicted)

            logging.info(f"Accuracy Score: {accuracy}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")
            logging.info(f"Classification Report:\n{class_report}")

            return {
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix,
                "classification_report": class_report
            }
        
        except Exception as e:
            raise Custom_Exception(e, sys)
