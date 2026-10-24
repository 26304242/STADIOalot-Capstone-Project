# Model 1 — Logistic Regression

**Notebook:** `Scripts/Models/model1_logistic_regression.ipynb`

The notebook reads `engineered_dataset.csv` or `Datasets/Engineered/engineered_dataset.csv`. It excludes `returned` and `order_id` from predictors, performs an 80/20 stratified split with `random_state=42`, imputes numeric/categorical values inside a pipeline, one-hot encodes categorical variables, standardises numeric variables, and fits Logistic Regression.

**Hyperparameters:** `C=1.0`, `penalty='l2'`, `solver='liblinear'`, `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.

## How to run
Load the model1_logistic_regression.ipynb script into the directory

Run preprocessing and feature engineering first, then run this notebook from top to bottom.

If the repository layout is cloned directly: the input and output data paths can be adjusted accoring to the instructions in the scripts

if necessary, upload the engineered_dataset.csv into the same directory before running this script.