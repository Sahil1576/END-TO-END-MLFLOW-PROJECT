import os
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from pathlib import Path

from LogHandler.logger import logging
from LogHandler.custome_exception import CustomException

@dataclass
class DataIngestionConfig:

    def __init__(self):

        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        ingestion_config = config["data_ingestion"]

        self.fake_data_path = ingestion_config["fake_data_path"]
        self.true_data_path = ingestion_config["true_data_path"]

        self.test_size = ingestion_config["test_size"]

        self.raw_data_path = ingestion_config['raw_data_path']

        self.train_data_path = ingestion_config["train_data_path"]
        self.test_data_path = ingestion_config["test_data_path"]


class DataIngestion:

    def __init__(self):

        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:

            logging.info("Data ingestion started")

            # Read Fake News Dataset
            fake_df = pd.read_csv(self.config.fake_data_path)

            logging.info("Fake dataset loaded successfully")

            # Read Real News Dataset
            true_df = pd.read_csv(self.config.true_data_path)

            logging.info("Real dataset loaded successfully")

            # Assign labels
            fake_df["label"] = 0
            true_df["label"] = 1

            # Combine datasets
            df = pd.concat(
                [fake_df, true_df],
                axis=0,
                ignore_index=True
            )

            logging.info("Fake and Real datasets combined")



            # Save raw combined data
            df.to_csv(
                self.config.raw_data_path,
                index=False
            )

            logging.info("Combined raw dataset saved")

            # Split data
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=42,
                stratify=df["label"]
            )

            os.makedirs(
                "./data/raw",
                exist_ok=True
            )

            # Save train and test
            train_df.to_csv(
                self.config.train_data_path,
                index=False
            )

            test_df.to_csv(
                self.config.test_data_path,
                index=False
            )

            logging.info("Train and test data saved")

            logging.info(
                f"Training data shape: {train_df.shape}"
            )

            logging.info(
                f"Testing data shape: {test_df.shape}"
            )

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:

            logging.exception("Error occurred during data ingestion")

            raise CustomException(e)
        

DI = DataIngestion()
DI.initiate_data_ingestion()