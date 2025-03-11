import os
import sys
import uuid
from datetime import datetime
from threading import Thread
from thyroid_detection.config.configuration import Configuration
from thyroid_detection.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from thyroid_detection.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from thyroid_detection.component.data_ingestion import DataIngestion
from thyroid_detection.component.data_validation import DataValidation
from thyroid_detection.component.data_transformation import DataTransformation
from thyroid_detection.exception import AppException
from thyroid_detection.logger import logging


class Pipeline(Thread):
    def __init__(self, config: Configuration= Configuration()) -> None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            super().__init__(daemon=False, name="pipeline")
            self.config = config
        except Exception as e:
            raise AppException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise AppException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) \
            -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact
                                             )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise AppException(e, sys) from e
        
    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact) \
                                    -> DataTransformationArtifact:
           try:
                data_transformation = DataTransformation(
                                     data_transformation_config=self.config.get_data_transformation_config(),
                                     data_ingestion_artifact=data_ingestion_artifact,
                                     data_validation_artifact=data_validation_artifact
                                     )
                return data_transformation.initiate_data_transformation()
           except Exception as e:
                raise AppException(e, sys)    

    def run_pipeline(self):
        try:
            logging.info("Pipeline starting: Data Ingestion Phase.")
            experiment_id = str(uuid.uuid4())
            logging.info(f"Experiment ID: {experiment_id}")
            
            logging.info("Initiating Data Ingestion...")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Data ingestion completed. Train file at: {data_ingestion_artifact.train_file_path}, Test file at: {data_ingestion_artifact.test_file_path}")

            logging.info("Initiating Data Validation...")
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Data Validation completed.")

            logging.info("Initiating Data Transformation...")
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
                )
            logging.info("Data Transformation completed.")            

            logging.info("Pipeline completed till CSV generation.")
        except Exception as e:
            raise AppException(e, sys)
