# Explainable ML Assignment

## Overview

This project is an explainable ML model evaluation and recommendation platform
covering classical ML, boosting, deep learning, transformer fine-tuning,
explainability, and an agentic recommendation review. It includes notebooks,
reusable Python modules, a FastAPI service, Docker Compose orchestration, and
an Azure deployment design.

Dataset A, the breast-cancer data set, is used strictly as a technical
classification benchmark and never as a clinical or diagnostic claim.

## Repository Structure

    explainable-ml-assignment/
    |-- notebooks/    Part B through H walkthroughs and artifact generation
    |-- src/          reusable ML, API, agent, training, and utility modules
    |-- tests/        pytest coverage for utilities, models, agents, and API
    |-- artifacts/    generated model, metrics, and review outputs
    `-- cloud/        Azure deployment design documentation

## Setup

Python 3.11 is required.

1. Clone the repository:

       git clone https://example.com/your-org/explainable-ml-assignment.git
       cd explainable-ml-assignment

2. Create and activate a virtual environment:

       python -m venv venv
       source venv/bin/activate

   On Windows PowerShell:

       python -m venv venv
       .\venv\Scripts\Activate.ps1

3. Install dependencies:

       pip install -r requirements.txt

4. Copy the environment template:

       cp .env.example .env

   On Windows PowerShell:

       Copy-Item .env.example .env

   The defaults work out of the box. No real secret is required because the
   recommendation chain defaults to a deterministic local mock.

## Running the Notebooks

Run the notebooks in this order:

1. 01_math_stats_optimization.ipynb demonstrates the Part B mathematical,
   statistical, optimization, probability, and loss-function foundations.
2. 02_classical_models.ipynb trains Decision Tree, Random Forest, Gaussian
   Naive Bayes, and SVM baselines and compares their metrics and business cost.
3. 03_boosting_and_pipelines.ipynb tunes boosting pipelines and produces
   artifacts/model.joblib and artifacts/metrics.json. Later steps require
   these artifacts.
4. 04_deep_learning.ipynb demonstrates a NumPy perceptron plus TensorFlow and
   PyTorch fundamentals.
5. 05_transformer_finetuning.ipynb performs resource-capped AG News
   transformer fine-tuning and writes lightweight label and metric artifacts.
6. 06_explainability.ipynb produces SHAP and LIME summaries, including
   artifacts/shap_summary.json and artifacts/lime_summary.json. The agent
   requires these summaries.

## Running the Agent

After notebooks 03 and 06 have generated their required artifacts, run:

    python -m src.agent.workflow

The workflow performs guarded metrics and explainability review and writes the
final markdown report to artifacts/model_review.md.

## Running the API Locally

The API requires artifacts/model.joblib. Run notebook 03 first, or generate
the artifact with python -m src.train, then start Uvicorn:

    uvicorn src.api.main:app --reload

Available endpoints:

- GET /health reports API and model-load status.
- POST /predict returns a validated benchmark prediction and probability.
- POST /explain returns a prediction with five local SHAP contributions.
- GET /model-review returns generated review markdown or an empty
  not-yet-generated state.

## Running via Docker Compose

Run:

    docker compose up --build

Compose runs the trainer as a one-shot service, writes artifacts to the shared
model-artifacts volume, then starts the API only after the trainer exits with
status 0. A successful run leaves the API running, and
curl http://localhost:8000/health reports model_loaded: true.

## Running Tests

Run the full suite with:

    pytest

The repository currently contains seven test modules and 43 collected tests.
Use pytest -v for verbose test names and outcomes.

## Reproducing the Transformer Checkpoint

The fine-tuned transformer checkpoint from notebook 05 is not committed to
git because model checkpoints are too large for this repository. Re-run
notebooks/05_transformer_finetuning.ipynb end to end to reproduce the
checkpoint locally. That notebook also regenerates the lightweight
artifacts/label_mapping.json and artifacts/transformer_metrics.json outputs.

## Model & Dataset Summary

The project uses sklearn.datasets.load_breast_cancer with 30 numerical
features for binary classification. The saved metrics artifact currently names
XGBoost as the selected model from notebook 03's comparison. The Docker
trainer recreates that selected XGBoost pipeline with the tuned parameters
recorded from notebook 03, while the API loads the saved full pipeline once at
startup.

## Evaluation Approach

Models are compared with accuracy, precision, recall, F1, ROC-AUC, and a
confusion matrix. Selection also uses a business-cost calculation that weights
false negatives at 5 times the cost of false positives, so accuracy alone does
not determine the recommendation.
