from thyroid_detection.exception import AppException
import sys
import pandas as pd
from thyroid_detection.logger import logging
from typing import List
from thyroid_detection.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from thyroid_detection.entity.config_entity import ModelTrainerConfig
from thyroid_detection.util.util import load_numpy_array_data,save_object,load_object
from thyroid_detection.entity.model_factory import MetricInfoArtifact, ModelFactory,GridSearchedBestModel
from thyroid_detection.entity.model_factory import evaluate_classification_model


class ThyroidEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"




class ModelTrainer:

    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)
            print(transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info(f"Splitting training and testing input and target feature")
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]

            # **Fix: Convert NumPy arrays to Pandas DataFrame & Series**
            x_train = pd.DataFrame(x_train)
            y_train = pd.Series(y_train)
            x_test = pd.DataFrame(x_test)
            y_test = pd.Series(y_test)

            print('======== Model Trainer V2: x_train data info ======='*2)
            print(x_train.shape, y_train.shape)
            print (type(x_train), type(y_train))
            print("================================================================"*2) 

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_name

            logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)


            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")

            logging.info(f"Initiating operation model selecttion")
            best_model = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)
            print("Debugging: Output of get_best_model() ->", best_model)
            print("Number of values returned:", len(best_model))
            logging.info(f"Best model found on training dataset: {best_model}")

            # Evaluate the best model on training and validation data
            train_accuracy = best_model.best_model.score(x_train, y_train)
            test_accuracy = best_model.best_model.score(x_test, y_test)

            # Print Accuracy
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Validation Accuracy: {test_accuracy:.4f}")

            logging.info(f"Training Accuracy: {train_accuracy:.4f}")
            logging.info(f"Validation Accuracy: {test_accuracy:.4f}")

            # Check for Overfitting
            if train_accuracy - test_accuracy > 0.10:  # Difference > 10% indicates overfitting
                print("Model is Overfitting! Consider Regularization, More Data, or Early Stopping.")
            else:
                print("Model Generalizes Well.")

            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list:List[GridSearchedBestModel]=model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list ]

            ############################################# MODEL EVALUATION #######################################################################

            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info: MetricInfoArtifact = evaluate_classification_model(model_list=model_list,X_train=x_train,y_train=y_train,
                                                                   X_test=x_test,y_test=y_test,base_accuracy=base_accuracy
                                                                   )
            metric_info, model_scores = metric_info
            print(model_scores)
            ####################################################################################################################
           
            logging.info(f"Best found model on both training and testing dataset.")

            preprocessing_obj=  load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            print(type(metric_info))
            print(metric_info)

            if metric_info is not None:
                model_object = metric_info.model_object
            else:
                raise ValueError("No valid model found that meets the evaluation criteria.")


            trained_model_file_path=self.model_trainer_config.trained_model_dir
            print('======== Model Trainer: trained model file path ======='*2)
            print(trained_model_file_path)
            print("================================================================"*2) 


            thyroid_model = ThyroidEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_object)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path,obj=thyroid_model)


            model_trainer_artifact=  ModelTrainerArtifact(is_trained=True,message="Model Trained successfully",
                                            trained_model_file_path=trained_model_file_path,
                                            train_f1_weighted=metric_info.train_f1_weighted,
                                            test_f1_weighted=metric_info.test_f1_weighted,
                                            train_balanced_accuracy=metric_info.train_balanced_accuracy,
                                            test_balanced_accuracy=metric_info.test_balanced_accuracy,
                                            model_accuracy=metric_info.model_accuracy)


            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")