from setuptools import setup, find_packages

with open("README.md","r") as f:
    long_description=f.read()


REPO_NAME="END-TO-END-MLFLOW-PROJECT"

PROJECT_NAME="Basic_mlflow"
__version__ ="0.0.0"
AUTHOR="Sahil1576"
AUTHOR_EMAIL="sahilkatve7@gmail.com"
SRC_REPO="src"


setup(
    name=PROJECT_NAME,
    version=__version__,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR}/{REPO_NAME}",
    package_dir={"":SRC_REPO},
    packages=find_packages(where=SRC_REPO)
)