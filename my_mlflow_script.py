import dagshub
import mlflow
dagshub.init(repo_owner='Abhishek-guptaaa', repo_name='Credit-Card-Default-Prediction', mlflow=True)
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)