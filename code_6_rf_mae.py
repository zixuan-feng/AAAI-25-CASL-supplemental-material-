import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from functions import mixErr_mae_pct

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



#----------------- Lasso Algorithm -----------------                                                             
def Lasso_Alg(train_data,val_data,batch_num,lampara,mixpara, n_trees, max_depth):

    trainN = len(train_data)
    valN = len(val_data)
    
    # === Lasso ===
    train_data
    X, y = np.array(train_data[train_data.columns[1:]]), np.array(train_data['num_drug_death_t'])
    
    clf = RandomForestRegressor(criterion='absolute_error',n_estimators = n_trees, max_depth=max_depth)
    #clf = linear_model.Lasso(alpha=lampara)
    clf.fit(X, y)
    #betavals = clf.coef_
    #beta0val = clf.intercept_
    
    # === Training Results (t-1) [not relevant to cv] ===
    y_train_hat = clf.predict(np.array(train_data[train_data.columns[1:]]))    
    train_errS0, train_errS1, train_err_mix,count_S0,count_S1 = mixErr_mae_pct(trainN,mixpara,
                                                             np.array(train_data['num_drug_death_t']),
                                                             y_train_hat)
    print('train_err_mix',train_err_mix)
    
    # === Validation Results (t-1) ===
    
    y_val_hat = clf.predict(np.array(val_data[val_data.columns[1:]]))
    val_errS0, val_errS1, val_err_mix,count_S0,count_S1 = mixErr_mae_pct(valN,mixpara,
                                                       np.array(val_data['num_drug_death_t']),
                                                       y_val_hat)
    print('val_err_mix',val_err_mix)
    
    return val_err_mix, y_val_hat,val_errS0,val_errS1,count_S0,count_S1

#----------------- Cross Validation Algorithm -----------------                    

# for each set of parameters

def CV_5splits(batch_test_year,batch_num,
               lampara,mixpara,n_trees, max_depth):
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
        val_err_mix, val_ypred,fpr,mae,count_S0,count_S1 = Lasso_Alg(train_data,val_data,batch_num,
                                           lampara,mixpara,n_trees, max_depth)
        val_err_splits.append(val_err_mix)
        val_fpr_splits.append(fpr)
        val_mae_splits.append(mae)
        val_count_S0_splits.append(count_S0)
        val_count_S1_splits.append(count_S1)
    

    return val_err_splits, val_fpr_splits, val_mae_splits, val_count_S0_splits, val_count_S1_splits



def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def run_rf_grid_search(train_data, test_data, param_grid):
    results = []
    
    for params in ParameterGrid(param_grid):
        print(f"Running with params: {params}")
        

        n_trees = params['n_trees']
        max_depth = params['max_depth']
        mixpara = params['mixpara']
        

        val_err_mix, y_val_hat, val_errS0, val_errS1, val_count_S0, val_count_S1 = Lasso_Alg(
            train_data, test_data, batch_num, lampara=None, mixpara=mixpara, n_trees=n_trees, max_depth=max_depth)
        

        results.append({
            'n_trees': n_trees,
            'max_depth': max_depth,
            'mixpara': mixpara,
            'val_err_mix': val_err_mix,
            'fpr': val_errS0,  
            'mae': val_errS1,  
            'y_val_hat': y_val_hat.tolist()
        })
        

        save_results_to_csv(results, f"results_rf_grid_search.csv")

    print("Grid search complete. Results saved to results_rf_grid_search_2020.csv")

param_grid = {
    'n_trees': [100, 200, 500, 1000],  
    'max_depth': [10, 20, 50,100], 
    'mixpara': [100]  
}


batch_num = 2
batch_test_year = 2020


retrain_data, test_data = data_split_2batch(batch_test_year)


run_rf_grid_search(retrain_data, test_data, param_grid)

