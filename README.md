# Credit Card Default Prediction



import dagshub
dagshub.init(repo_owner='Abhishek-guptaaa', repo_name='Credit-Card-Default-Prediction', mlflow=True)
import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


this file run beform make my_mlflow.py then run main.app then my_mlflow.py run 

bash myflow ui

