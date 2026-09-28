# Preprocessing

**Notebook:** `Scripts/Feature Engineering and Preprocessing/preprocessing.ipynb`

The preprocessing notebook loads `raw_dataset.csv` or `Datasets/Raw/raw_dataset.csv`, checks missing values and duplicates, removes exact duplicate rows, validates the `returned` target and `order_id` identifier, and saves the cleaned result as `processed_dataset.csv` or `Datasets/Processed/processed_dataset.csv`.

Model-specific imputation, categorical encoding and scaling are not performed globally here. They are applied inside each model pipeline after the train/test split to reduce data leakage risk.

## How to run
Load the preprocessing.ipynb script and the raw_dataset.csv into the same directory
If the repository layout is cloned directly: the input and output data paths can be adjusted accoring to the instructions in the scripts
Run the notebook from top to bottom.
