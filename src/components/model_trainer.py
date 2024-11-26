import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split train and test input data")
            X_train, y_train, X_test, y_test = (
                # Store everything in X_train except the last column
                train_arr[:,:-1],
                # Store the last column in y_train
                train_arr[:,-1],
                # Store everything in X_test except the last column
                test_arr[:,:-1],
                # Store the last column in y_test
                test_arr[:,-1]
            )
            # Create a dictionary of models for training
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "Ridge": {},
                "Lasso": {},
                "XGBRegressor": {
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor": {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,
                                                 y_test=y_test, models=models, params=params)
            print(model_report)

            # Select the best r2 score from dict
            best_model_score = max(sorted(model_report.values()))
            # Retrieve the name of the model with the best r2 score from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Set an evaluation threshold that aligns with the business goal
            if best_model_score < 0.8:
                raise CustomException("No best model found")
            logging.info(f"Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2
        except Exception as e:
            raise CustomException(e, sys)
