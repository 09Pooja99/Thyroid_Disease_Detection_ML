import sys
import re
import os
import zipfile
import pandas as pd
import numpy as np
from six.moves import urllib # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit
from thyroid_detection.entity.config_entity import DataIngestionConfig
from thyroid_detection.entity.artifact_entity import DataIngestionArtifact
from thyroid_detection.exception import AppException
from thyroid_detection.logger import logging


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'=' * 20} Data Ingestion log started. {'=' * 20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise AppException(e, sys)

    def download_thyroid_data(self) -> str:
        try:
            # Extracting remote URL to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            # Folder location to download file
            zip_download_dir = self.data_ingestion_config.zip_download_dir

            if os.path.exists(zip_download_dir):
                os.remove(zip_download_dir)

            os.makedirs(zip_download_dir, exist_ok=True)

            thyroid_file_name = os.path.basename(download_url)

            zip_file_path = os.path.join(zip_download_dir, thyroid_file_name)

            logging.info(f"Downloading file from :[{download_url}] into :[{zip_file_path}]")
            urllib.request.urlretrieve(download_url, zip_file_path)
            logging.info(f"File :[{zip_file_path}] has been downloaded successfully.")
            return zip_file_path

        except Exception as e:
            raise AppException(e, sys) from e

    def extract_zip_file(self, zip_file_path: str) -> str:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            os.makedirs(raw_data_dir, exist_ok=True)

            logging.info(f"Extracting ZIP file: [{zip_file_path}] into dir: [{raw_data_dir}]")
            with zipfile.ZipFile(zip_file_path, 'r') as thyroid_zip_file_obj:
                thyroid_zip_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")

            # Return path of required file (allbp.data)
            allbp_file_path = os.path.join(raw_data_dir, "allbp.data")
            if not os.path.exists(allbp_file_path):
                raise AppException(f"File allbp.data not found in extracted contents", sys)

            return allbp_file_path

        except Exception as e:
            raise AppException(e, sys) from e

    def convert_data_to_csv(self, allbp_file_path: str) -> str:
        try:
            columns = [
                'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
                'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
                'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
                'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured',
                'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source', 'Class'
                ]
            df = pd.read_csv(allbp_file_path, delimiter=",", header=None, na_values="?", names=columns) 
            csv_file_path = os.path.splitext(allbp_file_path)[0] + ".csv"
            df.to_csv(csv_file_path, index=False)

            logging.info(f"Converted allbp.data to CSV at: [{csv_file_path}]")
            return csv_file_path

        except Exception as e:
            raise AppException(e, sys) from e

    def split_data_as_train_test(self, csv_file_path: str) -> DataIngestionArtifact:
        try:
           logging.info(f"Reading CSV file: [{csv_file_path}]")
           thyroid_data_frame = pd.read_csv(csv_file_path)

        # Data Cleaning and Preprocessing
           thyroid_data_frame.loc[thyroid_data_frame['age'] > 150, 'age'] = np.nan
           thyroid_data_frame.replace("?", pd.NA, inplace=True)
        
        # Remove unwanted symbols and numerics in target column (Class)
           thyroid_data_frame["Class"] = thyroid_data_frame["Class"].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).strip())
        
        # Drop unnecessary columns
           thyroid_data_frame.drop(columns=[
            "TBG", "TSH measured", "T3 measured", "TT4 measured", "T4U measured", "FTI measured", "TBG measured", "referral source"
              ], inplace=True)
        
        # Convert 'Class' Column into Readable Labels
           label_mapping = {
            'negative': 'negative',
            'increased binding protein': 'hyperthyroidism',
            'decreased binding protein': 'hypothyroidism'
            }
           thyroid_data_frame["Class"] = thyroid_data_frame["Class"].replace(label_mapping)

           logging.info(f"Splitting data into train and test")
           strat_train_set = None
           strat_test_set = None

           split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

           for train_index, test_index in split.split(thyroid_data_frame, thyroid_data_frame["Class"]):
             strat_train_set = thyroid_data_frame.loc[train_index]
             strat_test_set = thyroid_data_frame.loc[test_index]

           train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, "train.csv")
           test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, "test.csv")

           os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
           os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)

           logging.info(f"Exporting training dataset to file: [{train_file_path}]")
           strat_train_set.to_csv(train_file_path, index=False)

           logging.info(f"Exporting test dataset to file: [{test_file_path}]")
           strat_test_set.to_csv(test_file_path, index=False)

           data_ingestion_artifact = DataIngestionArtifact(
               train_file_path=train_file_path,
               test_file_path=test_file_path,
               is_ingested=True,
               message="Data ingestion completed successfully."
               )
           logging.info(f"Data Ingestion artifact: [{data_ingestion_artifact}]")
           return data_ingestion_artifact

        except Exception as e:
           raise AppException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            zip_file_path = self.download_thyroid_data()
            allbp_file_path = self.extract_zip_file(zip_file_path)
            csv_file_path = self.convert_data_to_csv(allbp_file_path)
            return self.split_data_as_train_test(csv_file_path)
        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20} Data Ingestion log completed. {'=' * 20} \n\n")
