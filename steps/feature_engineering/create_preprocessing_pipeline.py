from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from zenml import step
import pandas as pd

@step
def create_preprocessing_pipeline(dataset: pd.DataFrame, target_column: str) -> Pipeline:
    """
    Create a preprocessing pipeline for feature engineering.
    """
    dataset = dataset.drop([target_column],axis=1)
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    categorical_columns = dataset.select_dtypes(include='object').columns
    numerical_columns = dataset.select_dtypes(exclude='object').columns

    # YOUR CODE HERE
    # look into Pipeline of sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html 
    # create a Pipeline of steps of imputer and scaler with the corresponding names "imputer" and "scaler"
    numerical_transformer = Pipeline(steps=[('imputer',numerical_imputer),('scaler',scaler)])
    # create a Pipeline of steps of imputer and encoder with the corresponding names "imputer" and "encoder"
    categorical_transformer = Pipeline(steps=[
        ("imputer", categorical_imputer),
        ("encoder", encoder)
    ])
    # look into ColumnTransformer of sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    # create ColumnTransformer of numerical transformer named "num", and categorical transformer named "cat"
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ])
    
    full_pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor)
    ])
    
    return full_pipeline