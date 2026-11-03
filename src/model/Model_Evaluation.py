import os
import json
import yaml
import joblib
from scipy.sparse import load_npz
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from LogHandler.custome_exception import CustomException
from LogHandler.logger import logging

class DataEvalutionConfig:

    def __init__(self):
        with open("./config/config.yaml","r") as f:
            config = yaml.safe_load(f)

        data_evaluation = config['model_evaluation']

        self.X_test_path = data_evaluation['X_test_path']
        self.y_test_path = data_evaluation['y_test_path']

        self.load_model_path = data_evaluation['load_model_path']

        self.confusion_matrix_path = data_evaluation['confusion_matrix_path']
        self.evaluation_report_path = data_evaluation['evaluation_report_path']


class DataEvaluation:

    def __init__(self):

        self.config = DataEvalutionConfig()

    def generate_predictions(self, model, X_test):

        predictions = model.predict(X_test)

        return predictions

    def calculate_metrics(self, y_test, predictions):

        metrics = {

            "accuracy": accuracy_score(
                y_test, predictions
            ),

            "precision": precision_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ),

            "recall": recall_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ),

            "f1_macro": f1_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            )
        }

        return metrics

    def generate_classification_report(
        self,
        y_test,
        predictions
    ):

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            output_dict=True,
            zero_division=0
        )

        return report

    def create_confusion_matrix(
        self,
        y_test,
        predictions
    ):

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Fake", "Real"]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        ax.set_title("Fake News Detection - Confusion Matrix")

        fig.tight_layout()

        fig.savefig(
            self.config.confusion_matrix_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        return cm

    def save_evaluation_report(
        self,
        metrics,
        classification_report_data,
        confusion_matrix_data
    ):

        report = {

            "metrics": metrics,

            "classification_report":
                classification_report_data,

            "confusion_matrix":
                confusion_matrix_data.tolist()
        }

        with open(
            self.config.evaluation_report_path,
            "w"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        return report

    def initiate_data_evaluation(
        self,
        best_model_path,
        X_test_path,
        y_test_path
    ):

        try:

            logging.info("Model evaluation started")

            model = joblib.load(best_model_path)

            X_test = load_npz(X_test_path)
            y_test = np.load(y_test_path)

            # Generate predictions
            predictions = self.generate_predictions(
                model,
                X_test
            )

            # Calculate metrics
            metrics = self.calculate_metrics(
                y_test,
                predictions
            )

            # Classification report
            report = self.generate_classification_report(
                y_test,
                predictions
            )

            # Create confusion matrix
            cm = self.create_confusion_matrix(
                y_test,
                predictions
            )


            final_report = self.save_evaluation_report(
                metrics,
                report,
                cm
            )

            logging.info(
                f"Test Macro F1: {metrics['f1_macro']:.4f}"
            )

            logging.info("Model evaluation completed")

            return final_report

        except Exception as e:

            logging.exception("Model evaluation failed")
            raise CustomException(e) from e

DEC = DataEvalutionConfig()

DE = DataEvaluation()
DE.initiate_data_evaluation(DEC.load_model_path, DEC.X_test_path, DEC.y_test_path)