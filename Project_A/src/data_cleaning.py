import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
     Strategy for Preprocessing the Data
     """

    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            # col=['hotel,is_canceled','lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','meal','country','market_segment','distribution_channel','is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type','agent','company','days_in_waiting_list','customer_type','adr','required_car_parking_spaces','total_of_special_requests','reservation_status','reservation_status_date']
            numeric_columns = df.select_dtypes(include=['int', 'float']).columns
            df.drop(df.columns.difference(numeric_columns), axis=1, inplace=True)
            df.drop(['is_canceled', 'lead_time', 'adults', 'children', 'babies', 'previous_cancellations',
                     'previous_bookings_not_canceled', 'booking_changes', 'agent', 'adr', 'total_of_special_requests',
                     'company'], axis=1, inplace=True)
            df['required_car_parking_spaces'].fillna(float(0.5), inplace=True)
            df.dropna(how='any', inplace=True)
            return df
        except Exception as e:
            logging.error(f'Error in Pre-Processing the data: {e}')
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for Dividing Data into train and test
    """

    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Dividing Data into train and test
        """
        try:
            X = df.drop(['required_car_parking_spaces'], axis=1)
            y = df['required_car_parking_spaces']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in splitting the data:{}".format(e))
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle Data
        :return: Union[pd.DataFrame,pd.Series]
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in Handling Data:{}".format(e))
            raise e


# if __name__ == "__main__":
#     df = pd.read_csv("/Users/veedantbrahmbhatt/Pycharm/py/UnderstandMlops/Project_A/data/Hotel_Revenue.csv")
#     data_cleaning = DataCleaning(df, DataPreProcessStrategy())
#     data_cleaning.handle_data()
