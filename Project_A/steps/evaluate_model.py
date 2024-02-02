import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE,r2,RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: RegressorMixin,
        X_test:pd.DataFrame,
        y_test:pd.DataFrame)->Tuple[
    Annotated[float,"r2_score"],
    Annotated[float,"rmse"],
]:
    """
    Evaluates the model on Test Data
    :param X_test: Pandas DataFrame
           y_test: Pandas DataFrame
    :return: Evaluation scores r2 and RMSE
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculated_scores(y_test, prediction)
        mlflow.log_metric("mse",mse)
        r2_class = r2()
        r2_score = r2_class.calculated_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)
        rmse_class = RMSE()
        rmse = rmse_class.calculated_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        return r2_score, rmse
    except Exception as e:
        logging.error("Error in evaluating model: ()", format(e))
        raise e