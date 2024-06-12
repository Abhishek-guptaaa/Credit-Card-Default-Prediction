# import os
# import sys
# import pickle
# import pymysql
# import pandas as pd
# from dotenv import load_dotenv
# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV
# from src.mlproject.logger import logging
# from src.mlproject.exception import CustomException

# load_dotenv()

# host = os.getenv("host")
# user = os.getenv("user")
# password = os.getenv("password")
# db = os.getenv("db")

# def read_sql_data():
#     try:
#         mydb = pymysql.connect(
#             host=host,
#             user=user,
#             password=user,
#             db=db
#         )
#         logging.info(f"Connection Established: {mydb}")
#         df = pd.read_sql_query('select * from data', mydb)
#         print(df.head())

#         return df
        
#     except Exception as e:
#         logging.error(f"Error establishing connection: {e}")
#         raise CustomException(e, sys)



# def save_object(file_path, obj):
#     try:
#         with open(file_path, "wb") as file_obj:
#             pickle.dump(obj, file_obj)
#     except Exception as e:
#         raise CustomException(e, sys)


# def evaluate_models(X_train, y_train, X_test, y_test, models, param):
#     try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             model_name = list(models.keys())[i]  # Store the model name for better error messages
#             para = param[model_name]

#             gs = GridSearchCV(model, para, cv=3)
#             gs.fit(X_train, y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train, y_train)

#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)
#             test_model_score = r2_score(y_test, y_test_pred)

#             report[model_name] = test_model_score

#         return report

#     except KeyError as e:
#         logging.error(f"Key error: {e}. Check if the model names in the 'models' and 'param' dictionaries match.")
#         raise CustomException(e, sys)
#     except Exception as e:
#         logging.error(f"Error in evaluate_models: {e}")
#         raise CustomException(e, sys)




# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return pickle.load(file_obj)
#     except Exception as e:
#         raise CustomException(e, sys)

import os
import sys
import pickle
import pymysql
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.Credit_Fault.logger import logging
from src.Credit_Fault.exception import CustomException

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql_data():
    """Read data from SQL database."""
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=user,  # Check if password is correct, it's currently set to 'user'
            db=db
        )
        logging.info(f"Connection Established: {mydb}")
        df = pd.read_sql_query('select * from data', mydb)
        print(df.head())
        return df
        
    except Exception as e:
        logging.error(f"Error establishing connection: {e}")
        raise CustomException(e, sys)


def save_object(file_path, obj):
    """Save object to a file."""
    try:
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate machine learning models."""
    try:
        report = {}
        for model_name, model in models.items():
            gs = GridSearchCV(model, param[model_name], cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report

    except KeyError as e:
        logging.error(f"Key error: {e}. Check if the model names in the 'models' and 'param' dictionaries match.")
        raise CustomException(e, sys)
    except Exception as e:
        logging.error(f"Error in evaluate_models: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    """Load object from a file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)
