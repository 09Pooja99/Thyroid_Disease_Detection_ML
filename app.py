from flask import Flask 
from thyroid_detection.logger import logging
from thyroid_detection.exception import ThyroidException 
import sys

app= Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
        try:
              raise Exception("we are testing custom exception")
        except Exception as e:
            raise Exception(e,sys)
            logging.info(thyroid_detection.error_messege)
            logging.info("We are testing logging module")
        return "Machine Learning Project"

if __name__ == "__main__":
    app.run(debug= True)
