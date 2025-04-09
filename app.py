from flask import Flask, request
import sys
from thyroid_detection.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from thyroid_detection.logger import logging
from thyroid_detection.exception import AppException
import os, sys
import json
from thyroid_detection.config.configuration import Configuration
from thyroid_detection.constant import CONFIG_DIR, get_current_time_stamp
from thyroid_detection.pipeline.pipeline import Pipeline
from thyroid_detection.entity.thyroid_predictor import ThyroidPredictor, ThyroidData
from flask import send_file, abort, render_template
from thyroid_detection.logger import get_log_dataframe
import pandas as pd

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "thyroid_detection"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


THYROID_DATA_KEY = "housing_data"
THYROID_PREDICTION_VALUE_KEY = "thyroid_prediction_value"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'thyroid_detection'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    logging.info(f"req_path: {req_path}")
    
    os.makedirs("housing", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    logging.info(f"req_path: view_experiment_hist")
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    logging.info(f"req_path: train")
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        THYROID_DATA_KEY: None,
        THYROID_PREDICTION_VALUE_KEY: None
    }

    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        on_thyroxine = float(request.form['on_thyroxine'])
        query_on_thyroxine = float(request.form['query_on_thyroxine'])
        on_antithyroid_medication = float(request.form['on_antithyroid_medication'])
        sick = float(request.form['sick'])
        pregnant = float(request.form['pregnant'])
        thyroid_surgery = float(request.form['thyroid_surgery'])
        I131_treatment = float(request.form['I131_treatment'])
        query_hypothyroid = float(request.form['query_hypothyroid'])
        query_hyperthyroid = float(request.form['query_hyperthyroid'])
        lithium = float(request.form['lithium'])
        goitre = float(request.form['goitre'])
        tumor = float(request.form['tumor'])
        hypopituitary = float(request.form['hypopituitary'])
        psych = float(request.form["psych"])
        TSH = float(request.form['tsh'])
        T3 = float(request.form["t3"])
        TT4 = float(request.form["tt4"])
        T4U = float(request.form["t4u"])
        FTI = float(request.form["fti"])
        

        thyroid_data = ThyroidData(age=age,
                                   TSH= TSH,
                                   T3= T3,
                                   TT4= TT4,
                                   T4U=T4U,
                                   FTI=FTI,
                                   sex= sex,
                                   on_thyroxine= on_thyroxine,
                                   query_on_thyroxine= query_on_thyroxine,
                                   on_antithyroid_medication= on_antithyroid_medication,
                                   sick= sick,
                                   pregnant= pregnant,
                                   thyroid_surgery= thyroid_surgery,
                                   I131_treatment= I131_treatment,
                                   query_hypothyroid=query_hypothyroid,
                                   query_hyperthyroid=query_hyperthyroid,
                                   lithium= lithium,
                                   goitre=goitre,
                                   tumor=tumor,
                                   hypopituitary=hypopituitary,
                                   psych=psych
                                   )
        thyroid_df = thyroid_data.get_housing_input_data_frame()
        thyroid_predictor = ThyroidPredictor(model_dir=MODEL_DIR)
        thyroid_prediction = thyroid_predictor.predict(X=thyroid_df)
        context = {
            THYROID_DATA_KEY: thyroid_data.get_thyroid_data_as_dict(),
            THYROID_PREDICTION_VALUE_KEY: thyroid_prediction
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    logging.info(f"req_path: predict_bulk")
    
    try:
        if request.method == 'POST':
            file = request.files['file']
            thyroid_df = pd.read_csv(file)
            try:
                thyroid_df = thyroid_df.drop("Class", axis=1)
                #print(thyroid_df.columns)
                #print(thyroid_df.dtypes )
            except:
                pass
                
            thyroid_predictor = ThyroidPredictor(model_dir=MODEL_DIR)
            try:
                thyroid_prediction = thyroid_predictor.bulk_prediction(X=thyroid_df)
            except Exception as e:
                raise AppException(e, sys) from e
            
            thyroid_df["Predictions"] = thyroid_prediction
            thyroid_df["Predictions"] = thyroid_df["Predictions"]
            
            result = thyroid_df.to_html(show_dimensions=True,
                                         justify= "center",
                                         classes=['table', 'table-striped'],
                                         border=0.5,
                                         escape=True)

            message = "PREDICTION RESULT"
            
            return render_template("bulk_prediction.html", result=result, message= message)
        
        message = 'Something Is Wrong!'
        return render_template("bulk_prediction.html", result="", message= message)
    
    except Exception as e:
        raise AppException(e, sys) from e


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()