# This code project structure


import os
import logging 
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name="Credit_Fault"

list_of_files=[
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    "setup.py",
    "app.py",
    "requirements.txt",
    "train.py",
    "template/index.html"
    
    
]

for filepath in list_of_files:
    filepath=Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"creating empty file : {filepath}")
    
    else:
        logging.info(f"{filename} is already exists")

