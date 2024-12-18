Code and Data Package Accompanying the Manuscript "A Spatio-temporal Cluster-aware Supervised Learning Framework for Predicting County-level Drug Overdose Deaths"


Overview
--------
This repository contains data and code to develop and test models for predicting county-level drug overdose deaths. The model files in Python are used to perform grid searches for hyperparameter tuning, model testing for CASL, lasso regression, and random forest. A license for Gurobi 10.0.0 or higher will be required to replicate the study. The main components are:

- Processed and raw data files
- Python scripts for mode training and testing. 

Data File
----------
1. data_1_input_data_file.csv
   - Description: This file contains the processed dataset used for modeling and analysis. It includes all relevant features that have been pre-processed and engineered from the raw data. This is the input data file for all scripts. 

2. data_2_ct_names.csv
   - Description: This file contains the variable names for all cross terms, which will be used in applying the CASL model. 

Code Files for Model Training and Testing
----------

1. code_1_1_functions.py
   code_1_2_models.py
   - Descriptions: Python scripts containing all defined functions to implement the CASL model, which need to be imported for all CASL model-related scripts. 

2. code_2_CASL_grid_search.py
   - Description: Python script for performing a grid search over the hyperparameters of the CASL models for hyperparameter tuning. 

3. code_3_CASL_batch_testing.py
   - Description: Python script for batch testing of the CASL model. It automates the testing of the CASL model across multiple model setups.

4. code_4_CASL_testing.py
   - Description: Python script for testing CASL model with a specific set of hyperparameters and output the model parameters, clustering decisions, and prediction results. 

5. code_5_lasso_mae.py
   code_5_1_model_lasso.py
   - Description: Python script for training and evaluating a lasso regression model with loss function defined by MAE. 

6. code_6_rf_mae.py
   - Description: Python script for training and evaluating a random forest model with loss function defined by MAE. 

