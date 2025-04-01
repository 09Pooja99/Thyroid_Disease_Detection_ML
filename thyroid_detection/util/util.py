import yaml
from thyroid_detection.exception import AppException
import os,sys
from thyroid_detection.constant import *
import numpy as np
import dill
import pandas as pd


def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise AppException(e,sys)

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise AppException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise AppException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise AppException(e, sys) from e


def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise AppException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise AppException(e,sys) from e


def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        datatset_schema = read_yaml_file(schema_file_path)

        schema = datatset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path)

        error_messgae = ""


        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_messgae = f"{error_messgae} \nColumn: [{column}] is not in the schema."
        if len(error_messgae) > 0:
            raise Exception(error_messgae)
        return dataframe

    except Exception as e:
        raise AppException(e,sys) from e
    
def load_npz_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    """Load .npz file and validate columns against schema."""
    try:
        datatset_schema = read_yaml_file(schema_file_path)
        schema = datatset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        npz_data = np.load(file_path, allow_pickle=True)
        print(f"Loaded NPZ data type: {type(npz_data)}")
        print(f"NPZ array shape: {npz_data.shape}") 
        
        dataframe = pd.DataFrame(npz_data) 

        # Check if schema columns match
        error_message = ""
        expected_columns = list(schema.keys())

        if len(dataframe.columns) != len(expected_columns):
            error_message += f"Expected {len(expected_columns)} columns, but found {len(dataframe.columns)}.\n"

        for i, column in enumerate(expected_columns):
            if i >= len(dataframe.columns):  # Avoid index error
                error_message += f"\nMissing column: {column} from schema."
            else:
                dataframe.rename(columns={i: column}, inplace=True)  # Rename columns based on schema

        if error_message:
            raise Exception(error_message)

        return dataframe

    except Exception as e:
        raise AppException(e, sys) from e   