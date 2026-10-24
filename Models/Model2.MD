# Model 2 — Random Forest

**Notebook:** `Scripts/Models/model2_random_forest.ipynb`

The notebook reads `engineered_dataset.csv` or `Datasets/Engineered/engineered_dataset.csv` and uses the same 80/20 stratified split (`random_state=42`). Numeric values are median-imputed and categorical values are one-hot encoded. Numerical scaling is not used because Random Forest does not require it.

**Hyperparameters:** `n_estimators=300`, `max_features='sqrt'`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.

## How to run
Load the model2_random_forrest.ipynb script into the directory

Run preprocessing and feature engineering first, then run this notebook from top to bottom.

If the repository layout is cloned directly: the input and output data paths can be adjusted accoring to the instructions in the scripts

if necessary, upload the engineered_dataset.csv into the same directory before running this script.
