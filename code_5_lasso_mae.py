import pandas as pd
import numpy as np
from gurobipy import *
from functions import mixErr_mae_pct
from sklearn import linear_model
from code_5_1_model_lasso.py import Model_lasso_mae
from sklearn.model_selection import ParameterGrid

np.random.seed(123)

#----------------- Read Train & Val Data -----------------    
def data_split_2batch(batch_val_year):
    def data_two_years(df, t1, t2): # for training 
        df_yr = df[(df["year_t"] == t1) | (df["year_t"] == t2)]
        df_yr = df_yr.drop(columns = "year_t")
        return df_yr
    
    def data_one_year(df, t1): # for validation 
        df_yr = df[df["year_t"] == t1]
        df_yr = df_yr.drop(columns = "year_t")
        return df_yr
    
    df_all = pd.read_csv('data_1_input_data_file.csv')
    df_all = df_all.set_index('countyfips')
    
    train_data = data_two_years(df_all,batch_val_year-1,batch_val_year-2)
    val_data = data_one_year(df_all,batch_val_year)
    return train_data, val_data

'''
mixpara = 100
batch_num = 2
batch_test_year = 2021
retrain_data, test_data = data_split_2batch(batch_test_year)
valN = len(test_data)
X, y = np.array(retrain_data[retrain_data.columns[1:]]), np.array(retrain_data['num_drug_death_t'])
Xtest = np.array(test_data[test_data.columns[1:]])
'''

def lasso_predict(betavals, beta0val, X):
    N = len(X)
    y_hat = [0]*N
    mm = len(X[0])
    for i in range(N):
        y_hat[i] = sum(betavals[j]*X[i][j] for j in range(mm)) + beta0val
    return y_hat

#----------------- Lasso Algorithm -----------------                                                             
def Lasso_Alg(train_data,val_data,batch_num,lampara,mixpara):

    trainN = len(train_data)
    valN = len(val_data)
    
    # === Lasso ===
    train_data
    X, y = np.array(train_data[train_data.columns[1:]]), np.array(train_data['num_drug_death_t'])
    #clf = linear_model.Lasso(alpha=lampara)
    #clf.fit(X, y)
    #betavals = clf.coef_
    #beta0val = clf.intercept_
    betavals, beta0val = Model_lasso_mae(X, y, lampara)
    
    # === Training Results (t-1) [not relevant to cv] ===
    y_train_hat = lasso_predict(betavals, beta0val, 
                                np.array(train_data[train_data.columns[1:]]))    
    train_errS0, train_errS1, train_err_mix,count_S0,count_S1 = mixErr_mae_pct(trainN,mixpara,
                                                             np.array(train_data['num_drug_death_t']),
                                                             y_train_hat)
    print('train_err_mix',train_err_mix)
    
    # === Validation Results (t-1) ===
    
    y_val_hat = lasso_predict(betavals, beta0val, 
                              np.array(val_data[val_data.columns[1:]]))
    val_errS0, val_errS1, val_err_mix,count_S0,count_S1 = mixErr_mae_pct(valN,mixpara,
                                                       np.array(val_data['num_drug_death_t']),
                                                       y_val_hat)
    print('val_err_mix',val_err_mix)
    
    return val_err_mix, y_val_hat, betavals, beta0val,val_errS0,val_errS1,count_S0,count_S1

#----------------- Cross Validation Algorithm -----------------                    

# for each set of parameters

def CV_5splits(batch_test_year,batch_num,
               lampara,mixpara):
    val_err_splits = []
    val_fpr_splits =[]
    val_mae_splits =[]
    val_count_S0_splits =[]
    val_count_S1_splits =[]
    for i in range(5):
        # Step 1: generate data for each split given batch_test_year
        batch_val_year = batch_test_year - (i+1)
        print(batch_val_year)
        train_data, val_data = data_split_2batch(batch_val_year)
        
        # Step 2: call LASSO to get validation error for each split
        val_err_mix, val_ypred, val_beta, val_beta0,fpr,mae,count_S0,count_S1 = Lasso_Alg(train_data,val_data,batch_num,
                                                                lampara,mixpara)
        val_err_splits.append(val_err_mix)
        val_fpr_splits.append(fpr)
        val_mae_splits.append(mae)
        val_count_S0_splits.append(count_S0)
        val_count_S1_splits.append(count_S1)
        
    return val_err_splits, val_fpr_splits, val_mae_splits, val_count_S0_splits, val_count_S1_splits



def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def run_lasso_grid_search(train_data, test_data, param_grid):
    results = []
    
    for params in ParameterGrid(param_grid):
        print(f"Running with params: {params}")
        
        lampara = params['lampara']
        mixpara = params['mixpara']
        
        val_err_mix, y_val_hat, betavals, beta0val, val_errS0, val_errS1, val_count_S0, val_count_S1 = Lasso_Alg(
            train_data, test_data, batch_num, lampara, mixpara)
        
        try:
            betavals_list = betavals.tolist()
        except AttributeError:
            betavals_list = betavals  
        
        try:
            beta0val_list = beta0val.tolist()
        except AttributeError:
            beta0val_list = beta0val  
        
        results.append({
            'lampara': lampara,
            'mixpara': mixpara,
            'val_err_mix': val_err_mix,
            'fpr': val_errS0,  
            'mae': val_errS1  
            #'betavals': betavals_list,
            #'beta0val': beta0val_list,
            #'y_val_hat': y_val_hat.tolist()
        })
        
        save_results_to_csv(results, f"results_lasso_grid_search_2020.csv")

    print("Grid search complete. Results saved to results_lasso_grid_search.csv")

param_grid = {
    'lampara': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9, 1,2,5, 10],
    'mixpara': [100]
}

mixpara = 100
batch_num = 2
batch_test_year = 2020

retrain_data, test_data = data_split_2batch(batch_test_year)

run_lasso_grid_search(retrain_data, test_data, param_grid)

