import importlib
from typing import List
import numpy as np
import os
import sys
import pandas as pd
import yaml
from pyexpat import model
from cmath import log
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, log_loss
from collections import namedtuple
from thyroid_detection.logger import logging
from thyroid_detection.exception import AppException
from thyroid_detection.entity.config_entity import ModelTrainerConfig




GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"


InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel",
                                   ["model_serial_number","model_name", "model", "best_model", "best_parameters", "best_score"])

BestModel = namedtuple("BestModel",
                               ["model_serial_number", "model", "best_model", "best_parameters", "best_score"])


MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_f1_weighted", "test_f1_weighted",
                                 "train_balanced_accuracy", "test_balanced_accuracy",
                                 "model_accuracy", "index_number"])



def evaluate_classification_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float= 0.9)->MetricInfoArtifact:

    try:
        f1_train_list = []
        f1_test_list = []
        balanced_accuracy_train_list = []
        balanced_accuracy_test_list = []
        balanced_accuracy_diff_list = []
        log_loss_train_list = []
        log_loss_test_list = []
        roc_auc_ovr_weighted_test_list = []
        roc_auc_ovr_weighted_train_list= []
        model_name_list = []

        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            print(f"Models being evaluated: {[type(m).__name__ for m in model_list]}")
            model_name_list.append(type(model).__name__)

            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")

            #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train).astype(int)
            y_test_pred = model.predict(X_test).astype(int)

            # Calculating F1_weighted score on training and testing dataset
            train_f1 = f1_score(y_train, y_train_pred, average="weighted")
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")

            f1_train_list.append(train_f1)
            f1_test_list.append(test_f1)

            # Get number of unique classes
            num_classes = len(np.unique(y_train))
 

            # Convert predictions to one-hot encoded format
            y_train_pred_one_hot = np.eye(num_classes)[y_train_pred]
            y_test_pred_one_hot = np.eye(num_classes)[y_test_pred]

            # Calculating roc_auc_ovr_weighted

            roc_auc_ovr_weighted_train = roc_auc_score(y_train, y_train_pred_one_hot, multi_class='ovr', average='weighted')
            roc_auc_ovr_weighted_test = roc_auc_score(y_test, y_test_pred_one_hot, multi_class='ovr', average='weighted')
            roc_auc_ovr_weighted_train_list.append(roc_auc_ovr_weighted_train)
            roc_auc_ovr_weighted_test_list.append(roc_auc_ovr_weighted_test)



            # Calculating Balanced Accuracy on training and testing dataset
            train_balanced_accuracy_score = balanced_accuracy_score(y_train, y_train_pred)
            test_balanced_accuracy_score = balanced_accuracy_score(y_test, y_test_pred)

            balanced_accuracy_train_list.append(train_balanced_accuracy_score)
            balanced_accuracy_test_list.append(test_balanced_accuracy_score)
            denominator = train_balanced_accuracy_score + test_balanced_accuracy_score

            model_accuracy = (2 * (train_balanced_accuracy_score * test_balanced_accuracy_score)) / denominator if denominator != 0 else 0

            diff_test_train_acc = abs(test_balanced_accuracy_score - train_balanced_accuracy_score)
            balanced_accuracy_diff_list.append(diff_test_train_acc)


            # Fix log_loss calculation
            loss_train = log_loss(y_train, y_train_pred_one_hot)
            loss_test = log_loss(y_test, y_test_pred_one_hot)
            log_loss_test_list.append(loss_test)
            log_loss_train_list.append(loss_train)


            print("============ MODEL Factory:LOG LOSS=========="*2)
            print("loss_train: ", loss_train)
            print("loss_test: ", loss_test)
            print("================================"*2)

            ################################################################

            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train F1_weighted\t\t Test F1_weighted\t\t  roc_auc_ovr_weighted_train    \t\t roc_auc_ovr_weighted_train \t\t Train Balanced Accuracy Score\t\t Test Balanced Accuracy Score\t\t Average Score")
            logging.info(f"{train_f1}\t\t {test_f1}\t\t {roc_auc_ovr_weighted_train}\t\t {roc_auc_ovr_weighted_test}\t\t\t {train_balanced_accuracy_score}\t\t {test_balanced_accuracy_score} \t\t\t {model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].")
            logging.info(f"Log Loss Train: [{loss_train}].")
            logging.info(f"Log Loss Test: [{loss_test}].")

            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model

            logging.info(f"train balanced accuracy: {train_balanced_accuracy_score} and model_accuracy(average): {model_accuracy} and base_accuracy: {base_accuracy} and diff_test_train_accuracy{diff_test_train_acc}")

            f1_logic = (train_f1 >= 0.70) and abs(train_f1 - test_f1) <= 0.08
            roc_auc_logic = (roc_auc_ovr_weighted_train >= 0.85) and abs(roc_auc_ovr_weighted_train - roc_auc_ovr_weighted_test) <= 0.08
            model_accuracy_logic = (train_balanced_accuracy_score >= base_accuracy) and diff_test_train_acc <= 0.06
            loss_logic = (loss_train <= 1.0) and (loss_test <= 1.0)

            logging.info(f"train_f1: {train_f1}, test_f1: {test_f1}, abs(train_f1 - test_f1): {abs(train_f1 - test_f1)}")
            logging.info(f"roc_auc_ovr_weighted_train: {roc_auc_ovr_weighted_train}, roc_auc_ovr_weighted_test: {roc_auc_ovr_weighted_test}, abs(roc_auc_train - roc_auc_test): {abs(roc_auc_ovr_weighted_train - roc_auc_ovr_weighted_test)}")
            logging.info(f"train_bal_acc: {train_balanced_accuracy_score}, base_acc: {base_accuracy}, diff_test_train_acc: {diff_test_train_acc}")
            logging.info(f"loss_train: {loss_train}, loss_test: {loss_test}, abs(loss_train - loss_test): {abs(loss_train - loss_test)}")

            #if model_accuracy >= base_accuracy and diff_test_train_acc < 0.5:
            if f1_logic and roc_auc_logic and model_accuracy_logic and loss_logic:
                logging.info(f"Acceptable model found: {model_name}")
                print('='*20)
                print("TEST_1: MODEL IS ABOVE BASE ACCURACY")
                print('='*20)

                logging.info(f"model_accuracy: {model_accuracy} and base_accuracy: {base_accuracy} and diff_test_train_accuracy{diff_test_train_acc}")

                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                            model_object=model,
                                                            train_f1_weighted=train_f1,
                                                            test_f1_weighted=test_f1,
                                                            train_balanced_accuracy = train_balanced_accuracy_score,
                                                            test_balanced_accuracy= test_balanced_accuracy_score,
                                                            model_accuracy=model_accuracy,
                                                            index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            print('='*20)
            print("TEST_2: MODEL NOT ACCEPTED")
            logging.info(f"No model found with higher accuracy than base accuracy")

             #################################################################
        data = {"models": model_name_list,
                "f1_weighted_train": f1_train_list,
                "f1_weighted_test": f1_test_list,
                "roc_auc_ovr_weighted_train": roc_auc_ovr_weighted_train_list,
                "roc_auc_ovr_weighted_test": roc_auc_ovr_weighted_test_list,
                "balanced_accuracy_train": balanced_accuracy_train_list,
                "balanced_accuracy_test": balanced_accuracy_test_list,
                "balanced_accuracy_diff": balanced_accuracy_diff_list,
                "log_loss_train": log_loss_train_list,
                "log_loss_test": log_loss_test_list}
        df_scores=pd.DataFrame(data=data)
        print(f"df_scores: {df_scores}")
        return metric_info_artifact,df_scores
    except Exception as e:
        raise AppException(e, sys) from e


def get_sample_model_config_yaml_file(export_dir: str):
    try:
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,
                    "verbose": 1
                }

            },
            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY:
                        {"param_name1": "value1",
                         "param_name2": "value2",
                         },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }

                },
            }
        }
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise AppException(e, sys)


class ModelFactory:
    def __init__(self, model_config_path: str = None,):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)

            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")

            print(property_data)

            for key, value in property_data.items():
                logging.info(f"Executing: {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config:dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def class_for_name(module_name:str, class_name:str):
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            logging.info(f" Successfully imported module: {module_name}")

            if not hasattr(module, class_name):
                raise AttributeError(f" Class '{class_name}' not found in module '{module_name}'. Available classes: {dir(module)}")

            class_ref = getattr(module, class_name)
            logging.info(f"successfully retrieved class: {class_name} from module : {module_name}")
            return class_ref
        except Exception as e:           
            raise AppException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                             output_feature) -> GridSearchedBestModel:

        try:
            grid_search_cv_ref = ModelFactory.class_for_name(
                module_name=self.grid_search_cv_module,
                class_name=self.grid_search_class_name
                )                                                           

            grid_search_cv = grid_search_cv_ref(
                estimator=initialized_model.model,
                param_grid=initialized_model.param_grid_search
                )
            grid_search_cv = ModelFactory.update_property_of_class(
                grid_search_cv,
                self.grid_search_property_data
                )
            
            model_name = type(initialized_model.model).__name__

            logging.info("GridSearchCV fitting started.")
            grid_search_cv.fit(input_feature, output_feature)
            logging.info("GridSearchCV fitting completed successfully.")

            logging.info(f'{">>"* 30} Training {model_name} completed. {"<<"*30}')

            print("Grid Search Results:", grid_search_cv.cv_results_)


            if not hasattr(grid_search_cv, "best_estimator_") or grid_search_cv.best_estimator_ is None:
                logging.error(f"GridSearchCV did not find a best estimator for {model_name}.")
                raise ValueError(f"No best model found during GridSearchCV for {model_name}.")
            
            best_estimator = grid_search_cv.best_estimator_
            best_params = grid_search_cv.best_params_
            best_score = grid_search_cv.best_score_

            logging.info(f"Best Model: {best_estimator}")
            logging.info(f"Best Parameters: {best_params}")
            logging.info(f"Best Score: {best_score:.4f}")

            return GridSearchedBestModel(
                model_serial_number=initialized_model.model_serial_number,                                              
                model_name=str(initialized_model.model),
                model=initialized_model.model,
                best_model=best_estimator,
                best_parameters=best_params,
                best_score=best_score
                )
        except Exception as e:
            logging.error(f"Error in execute_grid_search_operation: {e}")
            raise AppException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        try:
            initialized_model_list = []
            for model_serial_number, model_config in self.models_initialization_config.items():
                try:
                    logging.info(f" Initializing model {model_serial_number}...")

                    if MODULE_KEY not in model_config or CLASS_KEY not in model_config:
                        logging.warning(f" Missing module or class key in config for model {model_serial_number}. Skipping...")
                        continue
                    model_obj_ref = ModelFactory.class_for_name(
                        module_name=model_config[MODULE_KEY],
                        class_name=model_config[CLASS_KEY]
                        )
                    model = model_obj_ref()
                    if PARAM_KEY in model_config:
                        model_properties = dict(model_config[PARAM_KEY])
                        model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                      property_data=model_properties)
                    param_grid_search = model_config.get(SEARCH_PARAM_GRID_KEY, {})
                    if not param_grid_search:
                        logging.warning(f" No hyperparameter grid provided for model {model_serial_number}. It may not be tuned properly.")
                    model_name = f"{model_config[MODULE_KEY]}.{model_config[CLASS_KEY]}"

                    initialized_model_detail = InitializedModelDetail(
                        model_serial_number=model_serial_number,
                        model=model,
                        param_grid_search=param_grid_search,
                        model_name=model_name
                        )

                    initialized_model_list.append(initialized_model_detail)
                    logging.info(f" Model {model_serial_number} ({model_name}) initialized successfully.")
                except Exception as model_error:
                    logging.error(f" Error initializing model {model_serial_number}: {model_error}")
                    continue
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            logging.error(f" Critical error in get_initialized_model_list: {e}")
            raise AppException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(
            self, initialized_model: InitializedModelDetail, input_feature, output_feature
            ) -> GridSearchedBestModel:
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(
            self,
            initialized_model_list: List[InitializedModelDetail],input_feature,output_feature
            ) -> List[GridSearchedBestModel]:
        try:
            if not initialized_model_list:
                logging.warning(" No initialized models provided for parameter search.")
                return []
            
            logging.info(f" Starting best parameter search for {len(initialized_model_list)} models.")
            self.grid_searched_best_model_list = []
            for initialized_model in initialized_model_list:
                try:
                    logging.info(f" Running grid search for model {initialized_model.model_name}...")
                    grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                        initialized_model = initialized_model,
                        input_feature=input_feature,
                        output_feature=output_feature
                        )
                    self.grid_searched_best_model_list.append(grid_searched_best_model)
                    logging.info(f"Grid search completed for {initialized_model.model_name}.")
                except Exception as model_error:
                    logging.error(f" Error during grid search for {initialized_model.model_name}: {model_error}")
                    continue
            return self.grid_searched_best_model_list
        except Exception as e:
            logging.error(f" Critical error in initiate_best_parameter_search_for_initialized_models: {e}")
            raise AppException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:

        try:
            model_data =next((model for model in model_details if model.model_serial_number==model_serial_number),None)
            if model_data is None:
                logging.warning(f"Model with serial number {model_serial_number} not found.")
                raise ValueError(f"Model with serial number {model_serial_number} not found.")
            return model_data
        except Exception as e:
            logging.error(f"Error in get_model_detail: {e}")
            raise AppException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(
        grid_searched_best_model_list: List[GridSearchedBestModel],
        base_accuracy: float = 0.9
        ) -> BestModel:
        try:
            best_model = max(
                (model for model in grid_searched_best_model_list if model.best_score > base_accuracy),
                key=lambda model: model.best_score,
                default=None
                )
            if not best_model:
                highest_score = max(model.best_score for model in grid_searched_best_model_list) if grid_searched_best_model_list else None
                logging.warning(f" No model met the base accuracy threshold of {base_accuracy}. Highest found: {highest_score}")
                raise ValueError(f"No model met the required accuracy threshold. Highest found: {highest_score}")
            logging.info(f"Best model selected: {best_model.model_name} with accuracy {best_model.best_score:.4f}")
            return best_model
        except Exception as e:
            logging.error(f" Error in get_best_model_from_grid_searched_best_model_list: {e}")
            raise AppException(e, sys) from e


    def get_best_model(self, X, y, base_accuracy =0.9) -> BestModel:
        try:
            # Step 1: Get initialized models
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
           
            # Step 2: Perform Grid Search            
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
                )

            # Step 3: Get the Best Model
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list,
                base_accuracy=base_accuracy
                )

        except Exception as e:
            raise AppException(e, sys) from e