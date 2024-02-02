from zenml.steps import BaseParameters
class ModelNameConfig(BaseParameters):
    model_name:str="LinearRegression"
    model_kwargs:dict={}