import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder , FunctionTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from thyroid_detection.exception import AppException
from thyroid_detection.logger import logging
from thyroid_detection.entity.config_entity import DataTransformationConfig
from thyroid_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from thyroid_detection.constant import *
from thyroid_detection.util.util import read_yaml_file, save_object, save_numpy_array_data, load_data


class DataTransformation:

    def __init__(self, data_transformation_config:DataTransformationConfig,
                  data_ingestion_artifact:DataIngestionArtifact,
                    data_validation_artifact:DataValidationArtifact
                    ):
        try:
            logging.info(f"{'=' * 20} Data Transformation log started. {'=' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path
            dataset_schema = read_yaml_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]

            scalers = {
                'age': StandardScaler(),
                'TSH': RobustScaler(),
                'TT4': RobustScaler(),
                'T4U': MinMaxScaler(),
                'FTI': MinMaxScaler(),
                'T3': RobustScaler()
                }

            num_pipeline_steps = [
                ('imputer', SimpleImputer(strategy="median")),
                ('log_transform_TSH', FunctionTransformer(lambda x: np.log1p(np.nan_to_num(x)), validate=False)),
                ]
            num_pipeline_steps.append(('fillna_TSH', SimpleImputer(strategy='mean')))
            
            for col, scaler in scalers.items():
                num_pipeline_steps.append((f'scaler_{col}', scaler))

            num_pipeline = Pipeline(steps=num_pipeline_steps)

            # Separate 'sex' from other categorical columns
            binary_columns = [col for col in categorical_columns if col != "sex"]
            sex_column = ["sex"]

            # Binary Mapping Pipeline (for t -> 1, f -> 0)
            binary_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('binary_mapper', FunctionTransformer(lambda X: pd.DataFrame(X).applymap(lambda x: 1 if x == 't' else 0), validate=False))
            ])

            # Label Encoding Pipeline (for 'sex' column)
            label_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('label_encoder', FunctionTransformer(lambda X: np.array([LabelEncoder().fit_transform(X[:, i]) for i in range(X.shape[1])]).T, validate=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}") 


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('binary_pipeline', binary_pipeline, binary_columns),
                ('label_pipeline', label_pipeline, sex_column)
                ])

            return preprocessing

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Loading training and test data.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path

            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            schema = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]


            print("Before dropping target:", train_df.columns)
            print("Data types in original train_df:", train_df.dtypes)

            print(train_df.isnull().sum())


            logging.info(f"Splitting features and target variables.")
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name] 

            # Print unique values in the target column before encoding
            print("Unique values in target column before encoding:", y_train.unique())



            # Apply Label Encoding on Target Column
            logging.info("Applying Label Encoding on target column.")
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train).ravel()
            y_test = label_encoder.transform(y_test).ravel()

            
            # Print the mapping of original labels to encoded values
            label_mapping = {original: encoded for encoded, original in enumerate(label_encoder.classes_)}
            print("Label Encoding Mapping:", label_mapping)

            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            X_train_arr=preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            print("after preprocessing NaN counts: ", np.isnan(X_train_arr).sum())
            # Find positions of NaN values (optional)
            print("after preprocessing: ", np.where(np.isnan(X_train_arr)))
            print("after preprocessing: ", type(X_train_arr))
            print("before SMOTE shape:",  X_train_arr.shape, y_train.shape)
            logging.info(f"Before SMOTE shape: X_train: {X_train_arr.shape}, y_train: {y_train.shape}")

            # Apply SMOTE to handle class imbalance
            logging.info("Applying SMOTE to handle class imbalance.")
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            X_train_arr, y_train = smote.fit_resample(X_train_arr, y_train)

            print("after SMOTE shape:",  X_train_arr.shape, y_train.shape)
            logging.info(f"After SMOTE shape: X_train: {X_train_arr.shape}, y_train: {y_train.shape}")



            train_arr = np.c_[ X_train_arr, np.array(y_train)]

            test_arr = np.c_[X_test_arr, np.array(y_test)]

            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            preprocessing_obj_file_name = self.data_transformation_config.preprocessed_object_file_name

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_name

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise AppException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")