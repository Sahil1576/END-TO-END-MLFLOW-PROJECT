### End-To-End ML & MLOps

Intelligent Resume–Job Matching System | End-to-End ML & MLOps

1. Project Overview

The Intelligent Resume–Job Matching System is an end-to-end Machine Learning application that predicts how well a candidate's resume matches a given job description.

The system accepts resume text and job description text as inputs, converts both texts into numerical vectors, calculates their semantic similarity using cosine similarity, extracts relevant skills with the help of an LLM, and generates additional skill-based features. These features are passed to a trained regression model to predict a resume–job match score.

The project covers the complete Machine Learning lifecycle, including Exploratory Data Analysis (EDA), baseline modeling, experiment tracking, model registration, automated ML pipelines, model retraining, model evaluation, monitoring, API development, frontend development, Docker containerization, and CI/CD deployment.

2. Project Objectives

- Build an intelligent resume–job matching system using Machine Learning and LLM-based skill extraction.
- Perform EDA and understand the dataset before training models.
- Establish a baseline model and improve performance through multiple experiments.
- Track model parameters, metrics, and artifacts using MLflow.
- Register the best-performing model in the MLflow Model Registry.
- Build a modular and reproducible ML pipeline using functions and classes.
- Automate the pipeline using DVC.
- Implement a retraining mechanism that evaluates new models before promoting them.
- Monitor model predictions, latency, errors, and model health.
- Develop a FastAPI backend and an HTML, CSS, and JavaScript frontend.
- Automate Docker image builds and application deployments using GitHub Actions.

3. Dataset and Exploratory Data Analysis

The project begins with a resume–job matching dataset containing resume text, job description text, skill information, and match-related scores.

An EDA notebook is created to understand the dataset and identify potential data quality issues.

EDA includes:

- Dataset shape, column names, and data types.
- Missing-value and duplicate-record analysis.
- Distribution of target match scores.
- Resume and job description text-length analysis.
- Skill distribution and skill overlap analysis.
- Correlation analysis between numerical features.
- Identification of outliers and potentially leaked features.
- Analysis of class or score imbalance, where applicable.

The findings from EDA are used to guide feature engineering and model selection.

4. Feature Engineering

The system generates three primary features from resume and job description text.

4.1 Text Vectorization

The resume text and job description text are converted into numerical representations using TF-IDF.

A shared TF-IDF vectorizer is fitted on the appropriate training text corpus. The same fitted vectorizer then transforms both resume text and job description text into vectors.

This ensures that both vectors use the same vocabulary and feature space.

4.2 Cosine Similarity Score

Cosine similarity measures the similarity between the resume vector and job description vector.

The resulting score represents how similar the two texts are in the TF-IDF feature space.

4.3 LLM-Based Skill Extraction

An LLM is used to extract structured information from the resume and job description, including:

- Candidate skills.
- Required job skills.
- Matched skills.
- Missing skills.

The extraction process uses a consistent output schema so that the same feature-generation logic can be applied during training and inference.

4.4 Skill Match Ratio

The skill match ratio measures the proportion of required job skills found in the candidate's resume.

Formula:

"Skill Match Ratio = Number of Matched Skills / Number of Required Job Skills"

4.5 Skill Gap Ratio

The skill gap ratio measures the proportion of required job skills missing from the candidate's resume.

Formula:

"Skill Gap Ratio = Number of Missing Skills / Number of Required Job Skills"

Both ratios are handled safely when the job description contains no extractable required skills.

4.6 Additional Features for Future Experiments

Additional features can be evaluated through controlled experiments, such as:

- Resume and job description text lengths.
- Number of required skills and candidate skills.
- Relevant experience overlap.
- Education requirement match.
- Keyword coverage ratio.
- Skill importance-weighted match ratio.
- Sentence embedding similarity.
- Job category and experience-level features.

Additional features are introduced only when they improve validation performance and do not introduce target leakage.

5. Baseline Model and Experiment Tracking

The first model establishes a baseline against which future experiments can be compared.

The baseline workflow includes:

1. Prepare training, validation, and test datasets.
2. Apply feature engineering and preprocessing.
3. Train an initial regression model.
4. Evaluate the model on validation and test data.
5. Log parameters, metrics, and model artifacts using MLflow.
6. Compare subsequent experiments against the baseline.

The regression metrics include:

- MAE: Mean Absolute Error.
- MSE: Mean Squared Error.
- RMSE: Root Mean Squared Error.
- R² Score: Coefficient of determination.

Model parameters such as the algorithm, number of estimators, maximum depth, learning rate, and other applicable hyperparameters are logged with MLflow.

Multiple experiments are conducted using suitable regression algorithms and hyperparameter configurations. The best candidate is selected based on validation performance and relevant model-quality checks.

The selected model is registered in the MLflow Model Registry.

6. Modular ML Pipeline

The project follows a modular architecture using Python functions and classes.

The pipeline consists of the following stages:

Data Ingestion

- Load the source dataset.
- Read new data when available.
- Store the ingested data in the appropriate artifact directory.

Data Validation

- Validate required columns and data types.
- Check missing values and duplicate records.
- Validate target values and acceptable ranges.
- Detect schema changes and invalid records.

Data Transformation

- Clean and normalize text.
- Extract skills using the configured LLM workflow.
- Fit and apply the TF-IDF vectorizer.
- Calculate cosine similarity and skill-based features.
- Prepare model-ready features.
- Persist preprocessing artifacts for inference and retraining.

Model Training

- Train baseline and candidate models.
- Track experiments and parameters using MLflow.
- Save model artifacts and training metadata.

Model Evaluation

- Calculate MAE, MSE, RMSE, and R².
- Compare candidate performance against the baseline.
- Select a candidate based on validation metrics and acceptance criteria.

Model Registration

- Register the selected model in MLflow Model Registry.
- Track model versions and metadata.
- Promote a candidate only after it passes the required checks.

7. DVC Pipeline Automation

DVC is used to define and execute reproducible pipeline stages through "dvc.yaml".

The pipeline stages include:

- Data ingestion.
- Data validation.
- Data transformation.
- Model training.
- Model evaluation.

Stage dependencies, parameters, outputs, and relevant artifacts are defined so that DVC can identify which stages need to be rerun when their inputs change.

The pipeline can be executed using:

dvc repro

This helps make experiments reproducible and reduces unnecessary recomputation when pipeline dependencies have not changed.

8. Automated Model Retraining

A dedicated retraining workflow handles newly available training data.

When new labeled resume–job records become available, the retraining process follows these steps:

1. Load the existing training data and newly collected data.
2. Validate both datasets and check schema compatibility.
3. Combine old and new records.
4. Remove invalid records and handle duplicates appropriately.
5. Split the combined data into training and validation datasets while preserving an independent test set.
6. Reuse the existing model's hyperparameters retrieved from MLflow.
7. Apply the same feature-engineering and preprocessing strategy.
8. Train a new candidate model using the combined training data.
9. Evaluate both the existing production model and the candidate model on the same untouched test set.
10. Compare MAE, MSE, RMSE, and R², along with other model-quality checks.
11. Register and promote the candidate only if it meets the configured acceptance criteria.
12. Otherwise, retain the existing production model.

For this regression problem, lower MAE, MSE, and RMSE generally indicate lower prediction error, while higher R² generally indicates a better fit. Model promotion also considers data quality, validation results, and operational requirements.

Important: The model's preprocessing artifacts must remain compatible with the trained model. If the TF-IDF vocabulary or feature schema changes during retraining, the corresponding preprocessing artifacts must be versioned and deployed together with the candidate model.

9. MLflow Model Registry

MLflow Model Registry is used to manage model versions and promotion.

The registry workflow includes:

- Registering the selected candidate model.
- Maintaining version history.
- Tracking model parameters, metrics, and artifacts.
- Associating registered versions with evaluation results.
- Promoting an approved model to the production alias.
- Retaining the previous production version when a candidate is rejected.

The application loads the currently approved model version rather than automatically loading an arbitrary newly trained model.

10. Model Health and Monitoring Dashboard

A separate monitoring component provides visibility into model and application behavior.

The monitoring dashboard includes:

- Current production model version.
- Model status and last successful training time.
- Total prediction requests.
- Average and recent predicted match scores.
- Prediction latency.
- Successful and failed prediction counts.
- Prediction-score distribution.
- Prediction history and timestamps.
- Data-quality and schema validation failures.
- Available model-performance metrics when actual labeled outcomes become available.

Prediction logs store relevant metadata, including timestamps, predicted scores, latency, model version, and request status.

Raw resumes and personally identifiable candidate information are not stored in monitoring logs by default.

Monitoring limitation: Without actual ground-truth scores, the system cannot calculate genuine live MAE, MSE, RMSE, or R² for incoming predictions. These metrics require labeled outcomes or a separately evaluated labeled dataset.

11. FastAPI Backend

FastAPI provides the backend API for the resume–job matching application.

The backend is responsible for:

- Accepting resume text and job description text.
- Validating request data.
- Running the configured skill-extraction workflow.
- Generating the required numerical features.
- Loading the approved model and preprocessing artifacts.
- Predicting the match score.
- Returning matched skills, missing skills, and other available analysis.
- Recording prediction metadata for monitoring.
- Handling errors and returning appropriate API responses.

The API can also expose health and readiness endpoints to help determine whether the application and model are available.

12. Frontend Development

The user interface is developed using HTML, CSS, and JavaScript.

The frontend allows users to:

- Enter resume text.
- Enter a job description.
- Submit a matching request.
- View the predicted match score.
- Review matched skills and missing skills.
- Understand the generated skill-gap analysis.
- View relevant model-generated insights returned by the backend.

The frontend communicates with FastAPI through HTTP requests and displays the API response in a user-friendly format.

13. Docker Containerization

Docker is used to package the application and its runtime dependencies.

The Docker image includes the required application code, Python dependencies, and configuration needed to run the service.

Containerization provides a consistent runtime environment across local development and deployment.

The project can be built and tested locally before publishing an updated image.

14. CI/CD with GitHub Actions

GitHub Actions automates the build and deployment workflow.

The intended workflow is:

1. A code change is committed and pushed to GitHub.
2. GitHub Actions triggers the configured workflow.
3. The workflow installs dependencies and performs configured validation or tests.
4. Docker builds a new application image.
5. The image is tagged and pushed to the configured container registry.
6. The deployment workflow updates or triggers a deployment of the application on Render.
7. The deployed application is checked using a health endpoint or other configured smoke tests.

The workflow uses GitHub Secrets or environment variables for credentials and API keys.

Deployment note: Automatic image builds and deployments require the relevant Docker registry, Render service, credentials, and deployment triggers to be configured correctly. A successful CI/CD run does not by itself guarantee that the deployed application is healthy.

15. Technology Stack

Component| Technologies
Programming language| Python
Data analysis| Pandas, NumPy, Matplotlib, Seaborn
Feature engineering| TF-IDF, cosine similarity, skill-based ratios
Machine Learning| Scikit-learn and suitable regression algorithms
LLM integration| LLM-based skill extraction
Experiment tracking| MLflow
Model registry| MLflow Model Registry
Pipeline orchestration| DVC
Backend API| FastAPI
Frontend| HTML, CSS, JavaScript
Containerization| Docker
CI/CD| GitHub Actions
Deployment| Render
Monitoring| Custom prediction and model-health dashboard
Version control| Git and GitHub

16. End-to-End Architecture

Dataset
   |
   v
EDA and Data Understanding
   |
   v
Data Ingestion
   |
   v
Data Validation
   |
   v
Text Processing and LLM Skill Extraction
   |
   v
TF-IDF Vectorization and Feature Engineering
   |
   v
Baseline Model and Experiments
   |
   v
MLflow Tracking and Evaluation
   |
   v
Model Registry
   |
   v
FastAPI Backend
   |
   v
HTML / CSS / JavaScript Frontend
   |
   v
Prediction Logging and Monitoring

Retraining workflow:

New Labeled Data
       |
       v
Validate and Combine with Old Data
       |
       v
Prepare Training and Evaluation Data
       |
       v
Retrain Candidate Model
       |
       v
Compare with Production Model
       |
       v
Pass Acceptance Criteria?
       |
   +---+---+
   |       |
  Yes      No
   |       |
   v       v
Register   Keep Existing
and Promote Model
   |
   v
Monitor Production

CI/CD workflow:

Git Commit and Push
        |
        v
GitHub Actions
        |
        v
Tests and Validation
        |
        v
Build Docker Image
        |
        v
Push Image to Registry
        |
        v
Trigger Render Deployment
        |
        v
Health Check

17. Expected Outcome

The final application aims to provide a complete, modular, and reproducible ML workflow for resume–job matching, from dataset exploration and feature engineering to model training, registry management, automated retraining, API serving, monitoring, and CI/CD deployment.

It demonstrates practical Machine Learning Engineering and MLOps concepts, including experiment tracking, model versioning, controlled model promotion, pipeline automation, API development, containerization, and production monitoring.

The predicted match score represents the model's estimate based on the training target and engineered features. It should not be interpreted as a guaranteed hiring probability unless the model has been trained and validated against appropriately defined hiring-outcome labels.