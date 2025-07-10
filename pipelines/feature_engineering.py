from zenml import pipeline
from steps.feature_engineering.data_loader import data_loader
from steps.feature_engineering.calculate_age import calculate_age
from steps.feature_engineering.create_preprocessing_pipeline import create_preprocessing_pipeline
from steps.feature_engineering.data_splitter import data_splitter
from steps.feature_engineering.feature_engineering_preprocessing import feature_engineering_preprocessing

@pipeline(enable_cache=False)
def feature_engineering_pipeline():
    """
    Executes the feature engineering pipeline.

    This function loads the dataset, calculates the age, creates a preprocessing pipeline,
    splits the data into training and testing sets, and performs feature engineering preprocessing.
    """
    dataset = data_loader("./data/football_train.csv")
    dataset = calculate_age(dataset)
    pipeline = create_preprocessing_pipeline(dataset, "market_value_in_eur")
    X_train, X_test, y_train, y_test = data_splitter(dataset, "market_value_in_eur")
    X_train, X_test, pipeline = feature_engineering_preprocessing(X_train, X_test, pipeline)