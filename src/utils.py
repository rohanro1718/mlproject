import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}

        # for i in range(len(list(models))):
        #     model = list(models.values())[i]
        #     para = param[list(models.keys())[i]]

        for model_name,model in models.items():
            para = param[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                scoring="r2"
                )
            gs.fit(X_train,y_train)

            best_model = gs.best_estimator_
            models[model_name] = best_model

            # model.set_params(**gs.best_params_)
            # model.fit(X_train,y_train)

            # model.fit(X_train,y_train)
        
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            # print(f"Model: {model_name}")
            # print(f"Best Parameters: {gs.best_params_}")
            # print(f"Training R2: {train_model_score:.4f}")
            # print(f"Testing R2: {test_model_score:.4f}")
            # print("-" * 40)

            report[model_name] = test_model_score  
    
        return report  


    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        return CustomException(e,sys)