import logging
from zenml import step
import pandas as pd
import numpy as np
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import DataCleaning, DataDivideStrategy,DataPreProcessStrategy

@step
def clean_data(df: pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],]:
    """
    Cleans Data and divides into train and test sets 

    :param df: raw data
    :return: X_train,X_test,y_train,y_test
    """
    try:
        process_strategy=DataPreProcessStrategy()
        data_cleaning=DataCleaning(df,process_strategy)
        processed_data=data_cleaning.handle_data()

        divide_strategy=DataDivideStrategy()
        data_cleaning=DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test=data_cleaning.handle_data()
        logging.info("Data Cleaning and Spliting Completed")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f'Error in cleaning and Spliting Data {e}')
        raise e


