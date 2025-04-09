import sys
import re
import os
import zipfile
import pandas as pd
import numpy as np
import shutil
from typing import List
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
                shutil.rmtree(zip_download_dir)

            os.makedirs(zip_download_dir, exist_ok=True)

            thyroid_file_name = os.path.basename(download_url)

            zip_file_path = os.path.join(zip_download_dir, thyroid_file_name)

            logging.info(f"Downloading file from :[{download_url}] into :[{zip_file_path}]")
            urllib.request.urlretrieve(download_url, zip_file_path)
            logging.info(f"File :[{zip_file_path}] has been downloaded successfully.")
            return zip_file_path

        except Exception as e:
            raise AppException(e, sys) from e

    def extract_zip_file(self, zip_file_path: str) -> List[str]:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                shutil.rmtree(raw_data_dir)

            os.makedirs(raw_data_dir, exist_ok=True)

            logging.info(f"Extracting ZIP file: [{zip_file_path}] into dir: [{raw_data_dir}]")

            required_files = [
                "allbp.data", "allhyper.data", "allhypo.data",
                "allrep.data", "dis.data", "sick.data"
                ]
            extracted_files = []

            with zipfile.ZipFile(zip_file_path, 'r') as thyroid_zip_file_obj:
                for file in required_files:
                    if file in thyroid_zip_file_obj.namelist():
                        thyroid_zip_file_obj.extract(file, path=raw_data_dir)
                        extracted_files.append(os.path.join(raw_data_dir, file))
                    else:
                        raise AppException(f"Required file {file} not found in ZIP archive.", sys)
            logging.info(f"Extraction completed")
            return extracted_files

        except Exception as e:
            raise AppException(e, sys) from e

    def convert_data_to_csv(self, extracted_files: List[str]) -> List[str]:
        try:
            columns = [
                'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication',
                'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
                'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
                'TSH measured', 'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured',
                'T4U', 'FTI measured', 'FTI', 'TBG measured', 'TBG', 'referral source', 'Class'
                ]
            
            csv_file_paths = []
            for file_path in extracted_files:
                df = pd.read_csv(file_path, delimiter=",", header=0, na_values="?", names=columns)
                csv_file_path = os.path.splitext(file_path)[0] + ".csv"
                df.to_csv(csv_file_path, index=False)

                logging.info(f"Converted {os.path.basename(file_path)} to CSV at: [{csv_file_path}]")
                csv_file_paths.append(csv_file_path)
            return csv_file_paths

        except Exception as e:
            raise AppException(e, sys) from e
        
    def merge_csv_files(self, csv_file_paths: List[str], master_csv_file: str = "merged_thyroid_data.csv") -> str:
        try:
            merged_df = pd.DataFrame()
            
            for file_path in csv_file_paths:
                df = pd.read_csv(file_path)
                merged_df = pd.concat([merged_df, df], ignore_index=True)

            # Save merged CSV in the same directory as raw files
            raw_data_dir = os.path.dirname(csv_file_paths[0])
            merged_csv_path = os.path.join(raw_data_dir, master_csv_file)

            # Save merged data
            merged_df.to_csv(merged_csv_path, index=False)
            logging.info(f"Merged CSV saved to: {merged_csv_path}")

            return merged_csv_path

        except Exception as e:
            raise AppException(e, sys) from e

    def split_data_as_train_test(self, master_csv_file: str) -> DataIngestionArtifact:
        try:
           logging.info(f"Reading CSV file: [{master_csv_file}]")
           thyroid_data_frame = pd.read_csv(master_csv_file)

        # Data Cleaning and Preprocessing
           thyroid_data_frame.loc[thyroid_data_frame['age'] > 150, 'age'] = np.nan
           thyroid_data_frame.replace("?", pd.NA, inplace=True)
        
        # Remove unwanted symbols and numerics in target column (Class)
           thyroid_data_frame["Class"] = thyroid_data_frame["Class"].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).strip())
        
        # Drop unnecessary columns
           thyroid_data_frame.drop(columns=[
            "TBG", "TSH measured", "T3 measured", "TT4 measured", "T4U measured", "FTI measured", "TBG measured", "referral source"
              ], inplace=True)

           logging.info(f"Splitting data into train and test")
           strat_train_set = None
           strat_test_set = None

           split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

           for train_index, test_index in split.split(thyroid_data_frame, thyroid_data_frame["Class"]):
              strat_train_set = thyroid_data_frame.loc[train_index]
              strat_test_set = thyroid_data_frame.loc[test_index]

           # Separate features and target
           X_train = strat_train_set.drop(["Class"], axis=1)
           y_train = strat_train_set["Class"]

           X_test = strat_test_set.drop(["Class"], axis=1)
           y_test = strat_test_set["Class"]

           # Save train and test data
           train_set = pd.concat([X_train, y_train], axis=1)
           test_set = pd.concat([X_test, y_test], axis=1)

           train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, "train.csv")
           test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, "test.csv")

           os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
           os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)

           logging.info(f"Exporting training dataset to file: [{train_file_path}]")
           train_set.to_csv(train_file_path, index=False)

           logging.info(f"Exporting test dataset to file: [{test_file_path}]")
           test_set.to_csv(test_file_path, index=False)


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
            extracted_files = self.extract_zip_file(zip_file_path)
            csv_file_paths = self.convert_data_to_csv(extracted_files)
            merged_csv_path = self.merge_csv_files(csv_file_paths)
            return self.split_data_as_train_test(merged_csv_path)
        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20} Data Ingestion log completed. {'=' * 20} \n\n")
