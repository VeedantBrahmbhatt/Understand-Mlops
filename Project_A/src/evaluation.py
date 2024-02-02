import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np


class Evaluation(ABC):
    """
    Abstract class defining stratergy for evaluation of our models
    """
    @abstractmethod
    def calculated_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculates the scores for the model
        :param y_true:
        :param y_pred:
        :return: None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy Mean Square Error (MSE)

    """
    def calculated_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse=mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in Calculating MSE {}".format(e))
            raise e

class r2(Evaluation):
    """
    Evaluation strategy using r2 score
    """
    def calculated_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating r2 score")
            r2=r2_score(y_true,y_pred)
            logging.info("R2 score is {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score")
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy using RMSE
    """
    def calculated_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("MSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in Calculating MSE {}".format(e))
            raise e
