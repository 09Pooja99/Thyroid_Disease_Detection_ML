from setuptools import setup
from typing import List

#Declaring variables for setup functions
PROJECT_NAME="thyroid-predictor"
VERSION="0.0.1"
AUTHOR="Pooja Pandey"
DESCRIPTION="This is Thyroid predictor app"
PACKAGES=["thyroid_detection"]
REQUIREMENT_FILE_NAME="requirements.txt"

def get_requirements_list()->List[str]:
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        return requirement_file.readlines().remove("-e .")





setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=PACKAGES,
    install_requires=get_requirements_list()
    )

if __name__=="__main__":
    print(get_requirements_list())
