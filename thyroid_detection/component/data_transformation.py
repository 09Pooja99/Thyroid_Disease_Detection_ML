import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from thyroid_detection.exception import AppException
from thyroid_detection.logger import logging
from thyroid_detection.entity.config_entity import DataTransformationConfig
from thyroid_detection.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from thyroid_detection.constant import *
from thyroid_detection.util.util import read_yaml_file, save_object, save_numpy_array_data, load_data

class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact):
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

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore", sparse=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
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

            #print("Before dropping target:", train_df.columns)
            #print("Data types in original train_df:", train_df.dtypes)


            logging.info(f"Splitting features and target variables.")
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            # Identify numerical and categorical columns
            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            #cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            #print("Columns in training dataset:", list(X_train.columns))

            cat_cols = ['sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 
                    'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid',
                    'query hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']
            missing_cols = [col for col in cat_cols if col not in X_train.columns]
            if missing_cols:
                print("Missing columns:", missing_cols)
            # Handle missing columns (either add them with default values or remove references)
            for col in missing_cols:
                 X_train[col] = X_train[col].astype(str).fillna("missing")
                 X_test[col] = X_test[col].astype(str).fillna("missing")


            # Handle missing values for numerical columns
            num_imputer = SimpleImputer(strategy="median")
            X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            X_test[num_cols] = num_imputer.transform(X_test[num_cols])

            # Handle missing values for categorical columns
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

            # Apply OneHotEncoder to categorical variables
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_train_encoded = encoder.fit_transform(X_train[cat_cols])
            X_test_encoded = encoder.transform(X_test[cat_cols])

            #print("Transformed feature names:", X_train.columns)


            # Convert encoded categorical data to DataFrame
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(cat_cols))
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(cat_cols))

            # Drop original categorical columns and concatenate encoded ones
            X_train = X_train.drop(columns=cat_cols).reset_index(drop=True)
            X_test = X_test.drop(columns=cat_cols).reset_index(drop=True)
            X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
            #print("Data types after encoding and concatenation:", X_train.dtypes)
            #print("Unique values in age column after encoding:", X_train['age'].unique())
            X_test = pd.concat([X_test, X_test_encoded_df], axis=1)

            # Log transform TSH
            X_train['TSH'] = np.log1p(X_train['TSH'])
            X_test['TSH'] = np.log1p(X_test['TSH'])

            #print("NaN values in TSH after log transformation (X_train):", X_train['TSH'].isnull().sum())
            #print("NaN values in TSH after log transformation (X_test):", X_test['TSH'].isnull().sum())
            
            X_train['TSH'].fillna(X_train['TSH'].mean(), inplace=True)
            X_test['TSH'].fillna(X_test['TSH'].mean(), inplace=True)

            #Explicitly convert all numerical columns to numeric, handling potential errors
            for col in num_cols:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                X_train[col].fillna(X_train[col].mean(), inplace=True)
                X_test[col].fillna(X_test[col].mean(), inplace=True)
                X_train[col] = X_train[col].astype(float)
                X_test[col] = X_test[col].astype(float)

            # Debugging prints for all numerical columns
            #for col in num_cols:
                 #unique_values = X_train[col].unique()
                 #print(f"Column: {col}")
                 #for value in unique_values[:10]: # Print first 10 values
                     #print(f"  Value: {value}, Repr: {repr(value)}")
                 #print("-" * 20)


            # Apply scaling
            scalers = {
                'age': StandardScaler(),
                'TSH': RobustScaler(),
                'TT4': RobustScaler(),
                'T4U': MinMaxScaler(),
                'FTI': MinMaxScaler(),
                'T3': RobustScaler()
                }
            for col, scaler in scalers.items():
                 #print(f"Scaling column: {col}")
                 #print(f"  Data type before scaling (X_train): {X_train[col].dtype}")
                 X_train[col] = scaler.fit_transform(X_train[[col]]).flatten()
                 #print(f"  Data type after scaling (X_train): {X_train[col].dtype}")
                 X_test[col] = scaler.transform(X_test[[col]]).flatten()
                 #print(f"  Data type after scaling (X_test): {X_test[col].dtype}")


            #print("Columns in X_train:", list(X_train.columns))
            # After the scaling loop
            #print("NaN values in X_train after scaling:", X_train.isnull().sum().sum())
            #print("NaN values in X_test after scaling:", X_test.isnull().sum().sum())

            logging.info(f"Unique values in y_train before resampling: {y_train.unique()}")
            logging.info(f"Data type of y_train before resampling: {y_train.dtype}")

            # Resampling using SMOTE and Random Over-Sampling
            logging.info(f"Applying SMOTE and Random Over-Sampling.")
            print("Before SMOTE/ROS")
            print(f"y_train unique values: {y_train.unique()}") #check unique values
            #print(f"y_train data type: {y_train.dtype}") #check data type
            #print(f"X_train data types: {X_train.dtypes}") #check data types
            print(f"y_train value counts:\n{y_train.value_counts()}") 

            # Print unique values of all columns in X_train
            for col in X_train.columns:
                print(f"Unique values of X_train['{col}']: {X_train[col].unique()}")

            print(f"Number of NaN values in X_train: {X_train.isna().sum().sum()}")
            print(f"Number of infinite values in X_train: {np.isinf(X_train).sum().sum()}")

            # Apply Label Encoding after SMOTE and RandomOverSampler 

            #smote = SMOTE(sampling_strategy={"1": "auto"}, random_state=42) 
            #ros = RandomOverSampler(sampling_strategy={"0": "auto"}, random_state=42)

            #smote = SMOTE(sampling_strategy={"hypothyroidism": 20, "hyperthyroidism": "auto"}, random_state=42) #Use original labels.
            ros = RandomOverSampler( random_state=42) #SMOTE handles both.

            #X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

            X_train_balanced = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_train_balanced = pd.DataFrame(y_resampled, columns=["Class"])
            #print("After SMOTE/ROS")

            df_train_resampled= pd.concat([X_resampled, y_resampled], axis=1)
            logging.info(f"Resampling completed. Shape after SMOTE+ROS: {X_train_balanced.shape}")

            # Apply Label Encoding after SMOTE and RandomOverSampler 
            #print("Before Label Encoder")
            #print(f"y_train unique values: {y_train_balanced.unique()}")  # Resampled labels
            label_encoder = LabelEncoder()
            df_train_resampled["Class"] = label_encoder.fit_transform(df_train_resampled["Class"])
            print("After label encoder")

            
            
            # Define paths to save transformed data
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            preprocessing_obj_file_name = self.data_transformation_config.preprocessed_object_file_name
            
            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")
            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path=transformed_train_file_path, array=X_train_balanced.to_numpy())
            save_numpy_array_data(file_path=transformed_test_file_path, array=X_test.to_numpy())
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_name
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                message="Data transformation successful.",
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20} Data Transformation log completed. {'=' * 20}\n\n")