# import os
# import sys
# import mlflow
# from urllib.parse import urlparse
# from dataclasses import dataclass
# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
# #from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.tree import DecisionTreeClassifier
# from src.Credit_Fault.utils import evaluate_models, save_object
# from src.Credit_Fault.logger import logging
# from src.Credit_Fault.exception import CustomException

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join('artifacts', 'model.pkl')

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()


#     def eval_metrics(self, actual, pred):
#             accuracy = accuracy_score(actual, pred)
#             precision = precision_score(actual, pred, average='weighted')
#             recall = recall_score(actual, pred, average='weighted')
#             f1 = f1_score(actual, pred, average='weighted')

#             return accuracy, precision, recall, f1

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1],
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1]
#             )
#             models = {
#                  "Random Forest": RandomForestClassifier(),
#                  "Decision Tree": DecisionTreeClassifier(),
#                  "Gradient Boosting": GradientBoostingClassifier(),
#                  "XGBClassifier": XGBClassifier(),
#                  "CatBoost Classifier": CatBoostClassifier(verbose=False),
#                  "AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME')
#             }

#             params = {
#                 "Decision Tree": {
#                     'criterion': ['gini', 'entropy'],
#                     'splitter': ['best', 'random'],
#                     'max_depth': [None, 10, 20, 30, 40, 50],
#                 },
#                  "Random Forest": {
#                      'n_estimators': [8, 16, 32, 64, 128, 256],
#                      'max_depth': [None, 10, 20, 30, 40, 50]
#                 },
#                 "Gradient Boosting": {
#                      'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                      'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
#                      'n_estimators': [8, 16, 32, 64, 128, 256],
#                      'max_depth': [3, 4, 5, 6, 7, 8]
#                 },
#                 "XGBClassifier": {
#                      'learning_rate': [0.1, 0.01, 0.05, 0.001],
#                      'n_estimators': [8, 16, 32, 64, 128, 256],
#                      'max_depth': [3, 4, 5, 6, 7, 8]
#                 },
#                 "CatBoost Classifier": {
#                      'depth': [6, 8, 10],
#                      'learning_rate': [0.01, 0.05, 0.1],
#                      'iterations': [30, 50, 100]
#                 },
#                 "AdaBoost Classifier": {
#                      'learning_rate': [0.1, 0.01, 0.5, 0.001],
#                      'n_estimators': [8, 16, 32, 64, 128, 256]
#                 }
#             }

#             model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
#                                                  models=models, param=params)
            
#             # To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             # To get best model name from dict
#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             # ------------------------------------------------------
#             # ------------------------------------------------------
#             # MLflow code  this is write after model running
#             print("This is the best mode:")
#             print(best_model)
            
#             model_names = list(params.keys())

#             actual_model=" "
#             for model in model_names:
#                 if best_model_name == model:
#                     actual_model =actual_model + model
            
#             best_params = params[actual_model]

#             mlflow.set_registry_uri("https://dagshub.com/Abhishek-guptaaa/Credit-Card-Default-Prediction.mlflow")
#             tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).schema

#             #MLflow

#             with mlflow.start_run():
#                 predicted_qualities=best_model.predict(X_test)
#                 (accuracy, precision, recall, f1)= self.eval_metrics(y_test, predicted_qualities)
#                 mlflow.log(best_params)

#                 mlflow.log_metric("accuracy", accuracy)
#                 mlflow.log_metric("precision", precision)
#                 mlflow.log_metric("recall",recall)
#                 mlflow.log_metric("f1",f1)

#                 if tracking_url_type_store != "file":
#                     mlflow.sklearn.log_model(best_model, "model", registerd_model_name=actual_model)
#                 else:
#                     mlflow.sklearn.log_model(best_model, "model")




#             # ------------------------------------------------------
#             # ------------------------------------------------------


#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted = best_model.predict(X_test)

#             accuracy = accuracy_score(y_test, predicted)
#             return accuracy
#         except Exception as e:
#             raise CustomException(e, sys)



import os
import sys
import mlflow
from urllib.parse import urlparse
from dataclasses import dataclass
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, precision, recall, f1

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

            # ------------------------------------------------------
            # ------------------------------------------------------
            # MLflow code - this is written after the model runs
            print("This is the best model:")
            print(best_model)

            actual_model = best_model_name  # Directly use the best model name without leading space

            best_params = params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/Abhishek-guptaaa/Credit-Card-Default-Prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # MLflow
            with mlflow.start_run():
                predicted_qualities = best_model.predict(X_test)
                accuracy, precision, recall, f1 = self.eval_metrics(y_test, predicted_qualities)
                mlflow.log_params(best_params)  # Use log_params to log parameters

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            # ------------------------------------------------------
            # ------------------------------------------------------

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)



