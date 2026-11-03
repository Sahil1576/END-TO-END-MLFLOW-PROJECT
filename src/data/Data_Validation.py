import os
import json
import pandas as pd

from pathlib import Path
from LogHandler.custome_exception import CustomException
from LogHandler.logger import logging
import yaml


class DataValidationConfig:

    def __init__(self):
        with open("./config/config.yaml",'r') as config:
            config = yaml.safe_load(config)

        data_validation = config['data_validation']

        self.train_data_path = data_validation['train_data_path']
        self.test_data_path = data_validation['test_data_path']

        self.valid_label = data_validation['valid_label']

        self.require_column_name = data_validation['require_column_name']

        self.report_path = data_validation['report_path']

class DataValidation:

    def __init__(self):

        self.config = DataValidationConfig()

    def check_columns(self, df):
        columns = df.columns.to_list()

        for col in columns:
            if col not in self.config.require_column_name:
                raise ValueError(
                    f"Invalid Column name : {col}"
                )

        return True

    def check_missing_values(self, df):

        missing = df.isnull().sum().to_dict()

        return missing

    def check_duplicates(self, df):

        return int(df.duplicated().sum())

    def validate_labels(self, df):

        labels = df["label"].unique().tolist()

        invalid_labels = [
            label for label in labels
            if label not in self.config.valid_label
        ]

        if invalid_labels:
            raise ValueError(
                f"Invalid labels found: {invalid_labels}"
            )

        return True

    def check_empty_text(self, df):

        empty_count = (
            df["detail_text"].fillna("").str.strip().eq("")
        ).sum()

        if empty_count > 0:
            raise ValueError(
                f"{empty_count} records have empty detail_text"
            )

        return int(empty_count)

    def initiate_data_validation(self, train_path, test_path):

        try:

            logging.info("Data validation started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            self.check_columns(train_df)
            self.check_columns(test_df)

            self.validate_labels(train_df)
            self.validate_labels(test_df)

            train_missing = self.check_missing_values(train_df)
            test_missing = self.check_missing_values(test_df)

            train_duplicates = self.check_duplicates(train_df)
            test_duplicates = self.check_duplicates(test_df)

            self.check_empty_text(train_df)
            self.check_empty_text(test_df)

            # Check duplicate articles across train and test
            train_text = (
                train_df["detail_text"].fillna("").astype(str)
            ).str.lower().str.strip()

            test_text = (
                test_df["detail_text"].fillna("").astype(str)
            ).str.lower().str.strip()

            report = {
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_missing_values": train_missing,
                "test_missing_values": test_missing,
                "train_duplicates": train_duplicates,
                "test_duplicates": test_duplicates,
                "train_label_distribution":
                    train_df["label"].value_counts().to_dict(),
                "test_label_distribution":
                    test_df["label"].value_counts().to_dict()
            }


            report_path = os.path.join(
                self.config.report_path,
                "validation_report.json"
            )

            with open(report_path, "w") as file:
                json.dump(report, file, indent=4, default=str)


            logging.info("Data validation completed successfully")

            return True

        except Exception as e:

            logging.exception("Data validation failed")
            raise CustomException(e) from e

DVC = DataValidationConfig()

DV = DataValidation()
DV.initiate_data_validation(DVC.train_data_path, DVC.test_data_path)