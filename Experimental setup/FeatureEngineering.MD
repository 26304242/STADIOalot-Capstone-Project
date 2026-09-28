# Feature Engineering

**Notebook:** `Scripts/Feature Engineering and Preprocessing/feature_engineering.ipynb`

The notebook reads `processed_dataset.csv` or `Datasets/Processed/processed_dataset.csv` and creates derived predictors: `discount_amount`, `net_price`, `views_per_minute`, `price_per_view`, `purchase_return_interaction`, `high_discount_flag`, and `delayed_delivery_flag`. These features are based only on predictor variables and do not use `returned` or `order_id`.

The output is saved as `engineered_dataset.csv` or `Datasets/Engineered/engineered_dataset.csv`.

## How to run
Load the feature_engineering.ipynb script and the processed_dataset.csv into the same directory
If the repository layout is cloned directly: the input and output data paths can be adjusted accoring to the instructions in the scripts
Run preprocessing first, then run this notebook from top to bottom.
