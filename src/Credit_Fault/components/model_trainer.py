import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from src.Credit_Fault.utils import evaluate_models, save_object
from src.Credit_Fault.logger import logging
from src.Credit_Fault.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

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
                 "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                 "Gradient Boosting": GradientBoostingClassifier(),
                 "XGBClassifier": XGBClassifier(),
                 "CatBoost Classifier": CatBoostClassifier(verbose=False),
                 "AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME')
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                },
                 "Random Forest": {
                     'n_estimators': [8, 16, 32, 64, 128, 256],
                     'max_depth': [None, 10, 20, 30, 40, 50]
                 },
                 "Gradient Boosting": {
                     'learning_rate': [0.1, 0.01, 0.05, 0.001],
                     'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                     'n_estimators': [8, 16, 32, 64, 128, 256],
                     'max_depth': [3, 4, 5, 6, 7, 8]
                 },
                  "XGBClassifier": {
                     'learning_rate': [0.1, 0.01, 0.05, 0.001],
                     'n_estimators': [8, 16, 32, 64, 128, 256],
                     'max_depth': [3, 4, 5, 6, 7, 8]
                 },
                 "CatBoost Classifier": {
                     'depth': [6, 8, 10],
                     'learning_rate': [0.01, 0.05, 0.1],
                     'iterations': [30, 50, 100]
                 },
                 "AdaBoost Classifier": {
                     'learning_rate': [0.1, 0.01, 0.5, 0.001],
                     'n_estimators': [8, 16, 32, 64, 128, 256]
                 }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # if best_model_score <0.6:
            #     raise CustomException("No best model found", sys.exc_info()[1])
            # logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)




