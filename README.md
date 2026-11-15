# ScamShield

ScamShield is a local fraud-detection workbench and a worked Unit 5 machine-learning practical. It compares a **CART decision tree**, a **Random Forest**, and **XGBoost** on the same chronological test set.

## Start the dashboard

Extract the project. With Python 3.13 installed, open PowerShell in the **ScamShield** folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:5000**. Stop the server with **Ctrl+C**. The bundled models make the dashboard ready to use after dependency installation. On macOS or Linux, use `.venv/bin/python` in place of the Windows executable.

1. Open **Models** to compare held-out average precision, accuracy, precision, recall, and F1. Change the matrix selector to inspect the errors made by each model.
2. Open **Assess transaction**, select a held-out example and a classifier, then click **Run assessment**.
3. Edit the amount, time or any PCA input and run another assessment. The recorded label applies only to the original held-out row. Example outcome names refer to the dashboard's default classifier.

## Open the executed notebook

`ScamShield_Guide.ipynb` includes saved output and plots. Open it in Jupyter or VS Code with the project's Python interpreter. To launch JupyterLab from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyter lab ScamShield_Guide.ipynb
```

The guide covers source data, exploratory graphs, features, leakage prevention, time ordering, weighting, model fitting, cutoff selection, precision–recall, confusion matrices, a CART visualization, global importance, and a local explanation with an additivity check.

## Rebuild the models

```powershell
.\.venv\Scripts\python.exe train.py --data data/archive.zip
```

The project accepts the original ZIP or an extracted `creditcard.csv`. Running the command replaces the artifacts. Restart the dashboard afterward. The notebook can also rebuild the artifacts using its `RETRAIN` setting.

## Project files

- `core.py` defines the feature schema, loading, duplicate handling, chronological split, scoring, and explanations.
- `train.py` configures and trains models, selects thresholds on validation, evaluates on test, and writes reusable artifacts.
- `app.py`, `templates/`, and `static/` serve the browser dashboard and scoring endpoint.
- `artifacts/` holds the three fitted models, the evaluation report, and the aligned held-out scores used for plotting.
- `data/archive.zip` is the supplied source archive.

## Reading the results

The default model is selected by **validation average precision**. Test results are a separate measurement. Alert thresholds maximize validation F1 and use the inclusive rule **score ≥ cutoff**. Score magnitudes are uncalibrated because the classifiers are trained with class weights.

`V1` through `V28` are anonymized principal components. They can be ranked or perturbed to demonstrate model behavior, but their original business meanings are undisclosed. The data span roughly two days and the project is an educational benchmark.

Dataset credit: [ULB Machine Learning Group and Worldline · Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

