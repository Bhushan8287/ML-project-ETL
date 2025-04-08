import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "SVR": SVR(),
                "XGBoostRegressor": XGBRegressor(),
                "KNN": KNeighborsRegressor()
            }

            params = {
                "LinearRegression": {
                    "fit_intercept": [True, False],
                },
                "Ridge": {
                    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "fit_intercept": [True, False],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sag"]
                },
                "Lasso": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
                    "fit_intercept": [True, False],
                    "selection": ["cyclic", "random"]
                },
                "ElasticNet": {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "fit_intercept": [True, False],
                    "selection": ["cyclic", "random"]
                },
                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                },
                "SVR": {
                    "kernel": ["linear", "rbf", "poly"],
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto"],
                    "degree": [2, 3, 4]  # used only if kernel='poly'
                },
                "XGBoostRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size": [20, 30, 40],
                    "p": [1, 2]  # 1: Manhattan, 2: Euclidean
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,
                                                 X_test=X_test, y_test=y_test, models=models,
                                                 params=params)

            ## Get best model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test, predicted)
            return r2score

        except Exception as e:
            raise CustomException(e, sys)
