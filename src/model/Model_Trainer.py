import numpy as np
import json
import joblib
import yaml
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

from LogHandler.custome_exception import CustomException
from LogHandler.logger import logging

class ModelTrainerConfig:

    def __init__(self):
        with open("./config/config.yaml",'r') as config:
            config = yaml.safe_load(config)

        model_trainer = config['model_training']

        self.random_state = model_trainer['random_state']
        self.max_iter = model_trainer['max_iter']

        self.models = model_trainer['models']

        self.X_train_path = model_trainer['X_train_path']
        self.X_val_path = model_trainer['X_val_path']

        self.y_train_path = model_trainer['y_train_path']
        self.y_val_path = model_trainer['y_val_path']

        self.model_save_path = model_trainer['model_save_path']
        self.model_report_path = model_trainer['model_report_path']


class ModelTrainer:

    def __init__(self):

        self.config = ModelTrainerConfig()

    def initialize_models(self):

        model_params = self.config.models

        models = {

            "LogisticRegression": LogisticRegression(
                C=model_params['LogisticRegression']['C'],
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            ),

            "LinearSVC": LinearSVC(
                C=model_params['LinearSVC']['C'],
                random_state=self.config.random_state
            ),

            "MultinomialNB": MultinomialNB(
                alpha=model_params['MultinomialNB']['alpha']
            )
        }

        return models

    def train_model(self, model, X_train, y_train):

        model.fit(X_train, y_train)

        return model

    def evaluate_model(self, model, X_val, y_val):

        predictions = model.predict(X_val)

        score = f1_score(
            y_val,
            predictions,
            average="macro"
        )

        return score

    def save_model(self, model):

        joblib.dump(
            model,
            self.config.model_save_path
        )

        logging.info("Best model saved successfully")

    def initiate_model_training(self, X_train_path ,X_val_path ,y_train_path ,y_val_path):

        try:

            logging.info("Model training started")

            X_train = load_npz(X_train_path)
            y_train = np.load(y_train_path)

            X_val = load_npz(X_val_path)
            y_val = np.load(y_val_path)

            models = self.initialize_models()

            model_scores = {}
            trained_models = {}

            for model_name, model in models.items():

                logging.info(
                    f"Training started for {model_name}"
                )

                trained_model = self.train_model(model, X_train, y_train)

                score = self.evaluate_model(
                    trained_model,
                    X_val,
                    y_val
                )

                model_scores[model_name] = float(score)

                trained_models[model_name] = trained_model

                logging.info(
                    f"{model_name} Macro F1: {score:.4f}"
                )

            # Select best model based on validation F1
            best_model_name = max(
                model_scores,
                key=model_scores.get
            )

            best_model = trained_models[best_model_name]

            self.save_model(best_model)

            with open(
                self.config.model_report_path,
                "w"
            ) as file:

                json.dump(
                    {
                        "model_scores": model_scores,
                        "best_model": best_model_name,
                        "best_validation_f1": model_scores[best_model_name]
                    },
                    file,
                    indent=4
                )

            logging.info(
                f"Best model selected: {best_model_name}"
            )

            logging.info("Model training completed")

            return best_model

        except Exception as e:

            logging.exception("Model training failed")
            raise CustomException(e) from e


MTC = ModelTrainerConfig()

MT = ModelTrainer()
MT.initiate_model_training(MTC.X_train_path, MTC.X_val_path, MTC.y_train_path, MTC.y_val_path)