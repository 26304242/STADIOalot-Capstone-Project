# CAP182 SS3 Part B: Client-data modelling

## Deliverables

- `SS3_PartB.ipynb` — notebook implementing the five marked sections: dataset loading, preprocessing, model training, validation, and results visualisation.
- `requirements.txt` — Python packages needed to run the notebook.

## Data files

Place the five CSV files in a `Datasets/Client Data/` folder beside the notebook. The notebook also checks the notebook's current folder and `/mnt/data` (for this working environment). It recognises the original upload names, including `table1_orders_transactions(1).csv` and `table2_customer_behaviour(1).csv`.

Expected files:

- `table1_orders_transactions.csv`
- `table2_customer_behaviour.csv`
- `table3_return_outcomes.csv`
- `table4_product_reference.csv`
- `table5_seller_reference.csv`

Do not publish client data in a public repository unless the client and institution explicitly permit it. If the data is sensitive, keep the CSVs local and commit only the notebook, README and requirements file.

## Run instructions

1. Create and activate a Python environment (Python 3.10+ recommended).
2. Install dependencies with `pip install -r requirements.txt`.
3. Open `SS3_PartB.ipynb` in Jupyter Notebook, JupyterLab or VS Code.
4. Ensure the CSVs are in `data/` or beside the notebook.
5. Run all cells from top to bottom and inspect the generated metrics and plots.

## Modelling choices

The target is `returned` from the Return Outcomes table. Logistic Regression is retained for continuity with the SS2 model comparison. The notebook uses stratified training, validation and test subsets, numeric median imputation and standardisation, categorical most-frequent imputation and one-hot encoding, all fitted inside a scikit-learn pipeline. It reports precision, recall, F1-score, ROC-AUC, average precision and a confusion matrix.

## Data-quality note

The Orders/Transactions table contains 35 exact duplicate rows; the notebook removes exact duplicates before joining. The seller reference has repeated rows for seller ID `SEL00001` with conflicting numeric attributes. The notebook aggregates numeric seller fields by mean and the seller type by mode so that the join is many-to-one. This is documented as a pragmatic treatment that should be confirmed with the data owner before production use.

## Suggested Git commit sequence

1. `Add client data loading and class distribution checks`
2. `Implement validated table joins and data quality checks`
3. `Add feature engineering and leakage-safe preprocessing`
4. `Train Logistic Regression pipeline`
5. `Add validation metrics and unseen test evaluation`
6. `Add model visualisations and notebook documentation`

Commit each component as it is completed; do not create empty commits solely to meet the count requirement.

## Video Link

https://www.youtube.com/watch?v=2KkRvDX3dpk
