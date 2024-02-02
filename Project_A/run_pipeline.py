from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__=="__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    data_path="/Users/veedantbrahmbhatt/Pycharm/py/UnderstandMlops/Project_A/data/Hotel_Revenue.csv"
    training_pipeline(data_path)
#mlflow ui --backend-store-uri "file:/Users/veedantbrahmbhatt/Library/Application Support/zenml/local_stores/4caa623f-5ef1-476c-94c4-4bca2f284829/mlruns"
