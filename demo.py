from thyroid_detection.pipeline.pipeline import Pipeline
from thyroid_detection.config.configuration import Configuration
from thyroid_detection.logger import logging
from thyroid_detection.exception import AppException
import os

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()
