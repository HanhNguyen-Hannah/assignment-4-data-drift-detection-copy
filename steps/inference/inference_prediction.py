from zenml import step
from typing_extensions import Annotated
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
@step
def inference_prediction(batch: pd.DataFrame, model: RandomForestRegressor, drift: bool) -> Annotated[pd.DataFrame, "predictions"]:
    """
    Perform inference on a batch of data using a trained model.
    Args:
        batch: batch of inference data
        model: trained RandomForestRegressor
        drift: whether drift is detected in this batch of data

    Returns:
        predictions: batch of predictions
    """
    # YOUR CODE HERE
    # predict a given batch data using the traiend model
    pres = model.predict(batch)
    # store predictions into batch["predictions"]
    batch['predictions'] = pres
    # store drift boolean into batch["drift"]
    batch['drift'] = drift
    
    return batch