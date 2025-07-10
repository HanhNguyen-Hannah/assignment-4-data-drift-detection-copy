from pipelines.feature_engineering import feature_engineering_pipeline
from pipelines.training import training_pipeline
from pipelines.inference import inference_pipeline

if __name__ == "__main__":
    feature_engineering_pipeline()
    training_pipeline()
    inference_pipeline()