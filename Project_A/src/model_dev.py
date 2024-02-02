import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self,X_train,y_train):
        """
        Trains the model
        :param X_train:
        :param y_train:
        :return: None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self,X_train,y_train, **kwargs):
        """
        Trains them model
        :param X_train:
        :param y_train:
        :return:
        """
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model training Completed")
            return reg
        except Exception as e:
            logging.error("Error Training the model:{}".format(e))
            raise e
