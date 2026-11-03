import os
import re
import html
import joblib
import numpy as np
import pandas as pd
import yaml

from scipy.sparse import save_npz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from LogHandler.custome_exception import CustomException
from LogHandler.logger import logging

class DataTransformationConfig:

    def __init__(self):
        with open("./config/config.yaml",'r') as config:
            config = yaml.safe_load(config)

        data_transformation = config['data_transformation']

        self.train_data_path = data_transformation['train_data_path']
        self.test_data_path = data_transformation['test_data_path']

        self.max_features = data_transformation['max_features']
        self.vectorizer_path = data_transformation['vectorizer_path']
        self.validation_size = data_transformation['validation_size']
        self.stopwords = data_transformation['stopwords']
        self.ngram_range = data_transformation['ngram_range']

        self.train_encode_data_path = data_transformation['train_encode_data_path']
        self.test_encode_data_path = data_transformation['test_encode_data_path']
        self.val_encode_data_path = data_transformation['val_encode_data_path']

        self.y_train = data_transformation['y_train']
        self.y_val = data_transformation['y_val']
        self.y_test = data_transformation['y_test']

class DataTransformation:

    def __init__(self):

        self.config = DataTransformationConfig()

    def clean_text(self, text):

        text = html.unescape(str(text))

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()

    def split_validation_data(self, train_df):

        train_data, val_data = train_test_split(
            train_df,
            test_size=self.config.validation_size,
            random_state=42,
            stratify=train_df["label"]
        )

        return train_data, val_data

    def transform_features(self, train_data, val_data, test_data):

        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=tuple(self.config.ngram_range),
            stop_words=self.config.stopwords,
            sublinear_tf=True
        )

        # Fit only on training data
        X_train = vectorizer.fit_transform(
            train_data["detail_text"]
        )

        # Only transform validation and test data
        X_val = vectorizer.transform(
            val_data["detail_text"]
        )

        X_test = vectorizer.transform(
            test_data["detail_text"]
        )

        return vectorizer, X_train, X_val, X_test

    def save_artifacts(
        self,
        vectorizer,
        X_train,
        X_val,
        X_test,
        train_data,
        val_data,
        test_data
    ):

        # Save vectorizer
        joblib.dump(
            vectorizer,
            self.config.vectorizer_path
        )

        # Save sparse matrices
        save_npz(
            f"{self.config.train_encode_data_path}/X_train.npz",
            X_train
        )

        save_npz(
            f"{self.config.val_encode_data_path}/X_val.npz",
            X_val
        )

        save_npz(
            f"{self.config.test_encode_data_path}/X_test.npz",
            X_test
        )

        # Save target arrays
        np.save(
            f"{self.config.y_train}/y_train.npy",
            train_data["label"].to_numpy()
        )

        np.save(
            f"{self.config.y_val}/y_val.npy",
            val_data["label"].to_numpy()
        )

        np.save(
            f"{self.config.y_test}/y_test.npy",
            test_data["label"].to_numpy()
        )

        return True

    def initiate_data_transformation(self, train_path, test_path):

        try:

            logging.info("Data transformation started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Create validation split
            train_data, val_data = self.split_validation_data(
                train_df
            )

            # TF-IDF transformation
            vectorizer, X_train, X_val, X_test = (
                self.transform_features(
                    train_data,
                    val_data,
                    test_df
                )
            )

            os.makedirs(
                "./data/encoding/",
                exist_ok=True
            )

            # Save artifacts
            self.save_artifacts(
                vectorizer,
                X_train,
                X_val,
                X_test,
                train_data,
                val_data,
                test_df
            )

            logging.info("Data transformation completed")

            return {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": train_data["label"].to_numpy(),
                "y_val": val_data["label"].to_numpy(),
                "y_test": test_df["label"].to_numpy()
            }

        except Exception as e:

            logging.exception("Data transformation failed")
            raise CustomException(e) from e

DTC = DataTransformationConfig()

DT = DataTransformation()
DT.initiate_data_transformation(DTC.train_data_path, DTC.test_data_path)