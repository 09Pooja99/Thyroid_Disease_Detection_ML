import os
import sys
import uuid
from datetime import datetime
from threading import Thread
from thyroid_detection.config.configuration import Configuration
from thyroid_detection.entity.config_entity import DataIngestionConfig
from thyroid_detection.entity.artifact_entity import DataIngestionArtifact
from thyroid_detection.component.data_ingestion import DataIngestion
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

    def run_pipeline(self):
        try:
            logging.info("Pipeline starting: Data Ingestion Phase.")
            experiment_id = str(uuid.uuid4())

            logging.info(f"Experiment ID: {experiment_id}")
            logging.info("Initiating Data Ingestion...")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Data ingestion completed. Train file at: {data_ingestion_artifact.train_file_path}, Test file at: {data_ingestion_artifact.test_file_path}")

            logging.info("Pipeline completed till CSV generation.")
        except Exception as e:
            raise AppException(e, sys)
