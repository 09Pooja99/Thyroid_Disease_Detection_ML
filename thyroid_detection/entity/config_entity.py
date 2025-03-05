from collections import namedtuple 

DataIngestionConfig = namedtuple("DataIngestionConfig", 
    ["dataset_download_url", "zip_download_dir", "raw_data_dir", "ingested_dir", "ingested_train_dir", "ingested_test_dir"])

DataValidationConfig = namedtuple("DataValidationConfig", 
    ["schema_file_path", "report_file_path", "report_page_file_path"])

DataTransformationConfig = namedtuple("DataTransformationConfig", 
    ["transformed_dir", "transformed_train_dir", "transformed_test_dir", "preprocessing_dir", "preprocessed_object_file_name"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", 
    ["trained_model_dir", "model_file_name", "base_accuracy", "model_config_dir", "model_config_file_name"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", 
    ["model_evaluation_file_name"])

ModelPusherConfig = namedtuple("ModelPusherConfig", 
    ["model_export_dir"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", 
    ["pipeline_name", "artifact_dir"])
