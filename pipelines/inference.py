from zenml import pipeline
from zenml.client import Client
from steps.inference.load_batch_data import load_batch_data
from steps.inference.inference_prediction import inference_prediction
from steps.inference.inference_preprocessing import inference_preprocessing
from steps.inference.drift_detection import drift_detection
from steps.feature_engineering.calculate_age import calculate_age

@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Runs the inference pipeline for making predictions on new data.
    """
    client = Client()
    model = client.get_artifact_version("model")
    preprocessing_pipeline = client.get_artifact_version("pipeline")
    train_dataset = client.get_artifact_version("x_train")
    batch_data = load_batch_data("./data/player_of_interest.csv")
    batch_data = calculate_age(batch_data, True)
    drift, report = drift_detection(train_dataset, batch_data)
    if drift:
        print("Drift detected")
        # You could add at this point a step to notify the user or handle the drift!
    batch_data = inference_preprocessing(batch_data, preprocessing_pipeline)
    predictions = inference_prediction(batch_data, model, drift)
    return predictions
    