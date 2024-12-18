import pandas as pd
import numpy as np
from gurobipy import *
from code_1_1_functions import ypred_clu_cty, mixErr_mae_pct
from code_1_2_models import Model2_L1, Model2_L1_ex_hinge_loss

np.random.seed(123)




def cluReg_adjust_fips(bC,mm,cpara,lampara,wpara0,wpara1,rhopara,Kclu,df_data,deltavar): # adjust cluster centroid and regressor // given delta
    Kclu = int(Kclu)    

    train_fips = np.array(df_data.index.unique())
    df_dataX = df_data[df_data.columns[1:]]
    X_train = np.array(df_data[df_data.columns[1:]])#.drop(columns = name_loc))
    y_train = np.array(df_data['num_drug_death_t'])
    X_impt = np.array(df_data[df_data.columns[1:]].drop(columns = name_ct))
    #print(X_train.shape,X_impt.shape)
    #X_impt = X_train
    
    bC = bC
    N = len(y_train)
    m_impt = len(X_impt[0])
    #print(N/batch_num,len(train_fips))
    N_cty = len(train_fips)
    
    ind_supp = [0]*N # indicator var for suppressed cty-year, each i \in [N]
    S0 = [] #suppressed
    S1 = [] #non-suppressed
    for i in range(N):
        if y_train[i]==0:
            ind_supp[i] = 1
            S0.append(i)
        else:
            S1.append(i)
    N0 = len(S0)
    N1 = len(S1)

    # center adjustement: cenvar // take average of X in each cluster - all years
    def adjust_cty(data, index_set, k, n):
        centroid_new = []
        #print(n,len(index_set))
        for ind in range(k):
            indices = [i for i in range(n) if index_set[i][ind] >= 0.5] 
            indices = np.array(indices)
            indices_cty = train_fips[indices]
            df_cty = df_dataX.loc[indices_cty]
            data_cty = np.array(df_cty)
            centroid_val = np.mean(data_cty, axis=0) # compute means
            centroid_new.append(centroid_val)
        return np.array(centroid_new)
    cenvar = adjust_cty(X_impt, deltavar, Kclu, N_cty)
    #print('cenvar',cenvar.shape)
    
    def objclu_cty(data, index_set, centroid_new, k, n, m):
        objCen = [0]*k
        for ind in range(k):
            # indices for ind cluster
            indices = [i for i in range(n) if index_set[i][ind] >= 0.5] 
            indices = np.array(indices)
            indices_cty = train_fips[indices]
            df_cty = df_dataX.loc[indices_cty]
            data_cty = np.array(df_cty)
            # compute obj val
            objVal = float(1/len(indices))*sum((data_cty[p][j]-centroid_new[ind][j])*(data_cty[p][j]-centroid_new[ind][j]) for p in range(len(data_cty)) for j in range(m))
            objCen[ind] = objVal
        return objCen
    objCen = objclu_cty(X_train, deltavar, cenvar, Kclu, N_cty, m_impt)
    
    
    # regression for each cluster: betavars, beta0var//solve beta individually using model2
    # use all features
    def objBeta_cty(bC,mm,cpara,lampara,wpara0,wpara1,X_train,y_train,deltavar, Kclu, n):
        betavars = [0]*Kclu
        beta0var = [0]*Kclu
        objBeta = [0]*Kclu
        for ind in range(Kclu):
            
            cpara_num = cpara[ind]
            lampara_num = lampara[ind]
            # data for ind cluster
            '''
            indices = [i for i in range(n) if deltavar[i][ind] >= 0.5]
            cluX = X_train[indices]
            cluY = y_train[indices]
            '''
            indices = [i for i in range(n) if deltavar[i][ind] >= 0.5] 
            indices = np.array(indices)
            indices_cty = train_fips[indices]
            df_cty_Xy = df_data.loc[indices_cty]
            cluX = np.array(df_cty_Xy[df_cty_Xy.columns[1:]])
            cluY = np.array(df_cty_Xy['num_drug_death_t'])
            # do regression using Model2
            objVal,betavals,beta0val = Model2_L1_ex_hinge_loss(bC,mm,cpara_num,lampara_num,wpara0,wpara1,cluX,cluY,N0,N1,Kclu)
            betavars[ind] = betavals
            beta0var[ind] = beta0val
            objBeta[ind] = objVal
        return objBeta,betavars,beta0var
    objBeta,betavars,beta0var = objBeta_cty(bC,mm,cpara,lampara,wpara0,wpara1,X_train,y_train,deltavar, Kclu, N_cty)
    # print('objBeta',objBeta)
    # print('objCen',objCen)
    objVal_adjust = sum(objBeta) + rhopara*sum(objCen)
    return objVal_adjust,objBeta,objCen,cenvar,betavars,beta0var

def cluReg_assign_fips(bC,mm,cpara,lampara,wpara0,wpara1,rhopara,ratioClu,Kclu,df_data,betavars,beta0var,cenvar):
    Kclu = int(Kclu)
    batch_num = 2

    train_fips = np.array(df_data.index.unique())
    y_train = np.array(df_data['num_drug_death_t'])

    bC = bC
    N = len(y_train)
  
    #print(N/batch_num,len(train_fips))
    
    ind_supp = [0]*N # indicator var for suppressed cty-year, each i \in [N]
    S0 = [] #suppressed
    S1 = [] #non-suppressed
    for i in range(N):
        if y_train[i]==0:
            ind_supp[i] = 1
            S0.append(i)
        else:
            S1.append(i)
    N0 = len(S0)
    N1 = len(S1)

    def beta():
        betavars_lst = pd.DataFrame.from_dict(betavars).to_numpy()
        auxbetas = [abs(ele) for ele in betavars_lst]
        auxbeta0 = [abs(ele) for ele in beta0var]
        return auxbetas, auxbeta0
    #start = time.time()
    auxbetas, auxbeta0 = beta()
    auxbsum = sum((lampara[k]/Kclu)*(sum(auxbetas[k][j] for j in range(mm))+auxbeta0[k]) for k in range(Kclu))

    
    def aux_cty(batch_num, X_aux, y_aux, ind_supp_cty):
        S0_cty = [] 
        S1_cty = [] 
        for i in range(batch_num):
            if ind_supp_cty[i]>0:
                S0_cty.append(i)
            else:
                S1_cty.append(i)
                
        aux2 = [0]*Kclu
        for k in range(Kclu):
            a = [0.0]*batch_num
            for i in S0_cty:
                aux1 = float(sum(betavars[k][j]*X_aux[i][j] for j in range(mm))+beta0var[k]-cpara[k])
                #print(aux1)
                aux0 = float(max(0.0, aux1))
                a[i] = aux0
            aux2[k] = a
            
        aux3 = [0]*Kclu
        for k in range(Kclu):
            a = [0.0]*batch_num
            for i in S1_cty:            
                a[i] = abs(sum(betavars[k][j]*X_aux[i][j] for j in range(mm))+beta0var[k]-y_aux[i]) 
            aux3[k] = a
            
        return aux2, aux3
    
    def dist_cty(batch_num, X_dist):
        auxdissum = [0]*Kclu
        mm_dist = len(X_dist[0])
        for k in range(Kclu):
            a = [0.0]*batch_num
            for i in range(batch_num):
                a[i] = round(100*sum((X_dist[i][j] - cenvar[k][j])*(X_dist[i][j] - cenvar[k][j]) for j in range(mm_dist)))/100
            auxdissum[k] = a
        return auxdissum
    
    def cty_err():
        # Xy for each cty
        cty_err_tot = []
        for i in train_fips:
            df_cty = df_data.loc[i]
            X_cty = np.array(df_cty[df_cty.columns[1:]])
            y_cty = np.array(df_cty['num_drug_death_t'])
            
            ind_supp_cty = [1]*batch_num
            for j in range(batch_num):
                if y_cty[j]>0:
                    ind_supp_cty[j] = 0
            #print(y_cty,ind_supp_cty)
            
            aux2_cty, aux3_cty = aux_cty(batch_num,X_cty,y_cty,ind_supp_cty)
            auxdissum_cty = dist_cty(batch_num,X_cty)
            #print(len(auxdissum_cty))
            
            err_cty = [0]*Kclu
            for k in range(Kclu):
                err_cty[k] = sum(ind_supp_cty[i]*(wpara0/float(N0)*aux2_cty[k][i])
                                 +(1-ind_supp_cty[i])*(wpara1/float(N1)*aux3_cty[k][i])
                                 +rhopara/float(N)*auxdissum_cty[k][i] for i in range(batch_num))
            cty_err_tot.append(err_cty)
        return cty_err_tot
    cty_err_tot = cty_err()
    #print(cty_err_tot)
    #print('cty_err_tot',len(cty_err_tot),len(cty_err_tot[0]))
    
    def delta_cty(batch_num):
        N_cty = len(train_fips)
        #print('County num',N_cty)
        mb = Model()
        deltavar = mb.addVars(N_cty,Kclu, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='delta')
        mb.setObjective(sum(cty_err_tot[i][k]*deltavar[i,k] for i in range(N_cty) for k in range(Kclu)), GRB.MINIMIZE)
        mb.update()
        mb.addConstrs(sum(deltavar[i,k] for k in range(Kclu)) == 1 for i in range(N_cty))
        mb.addConstrs(sum(deltavar[i,k] for i in range(N_cty)) >= math.floor(ratioClu*N_cty/Kclu) for k in range(Kclu)) 
        mb.update()
    
        mb.params.OutputFlag = 0
        mb.params.timelimit = 300 
        mb.optimize()
        deltavar = [[deltavar[i,k].x for k in range(Kclu)] for i in range(N_cty)]
        return mb.objVal+auxbsum,deltavar
    
    obj,deltavar = delta_cty(batch_num)
    deltavar = np.array(deltavar)

    return obj, deltavar 
    #return mb.objVal+auxbsum,deltavar

            
#----------------- AM Algorithm -----------------                                                             
def AM_Alg(train_data,val_data,batch_num,wpara0,wpara1,rhopara,num_clu,ratioClu,lampara,mixpara,cpara):
    
    trainN = len(train_data)
    valN = len(val_data)
    mm = len(train_data.columns) - 1
    
    # === Initial Solution ===
    
    bC = 1e3
    betavals = np.random.uniform(0,1,(num_clu,mm))
    beta0val = np.random.uniform(0,1,num_clu)
    cenval = np.random.uniform(0,1,(num_clu,train_data.shape[1]))
    obj,deltaval = cluReg_assign_fips(bC,mm,cpara,lampara,wpara0,wpara1,rhopara,ratioClu,num_clu,
                                      train_data,betavals,beta0val,cenval)
    
    # === The AM Algorithm ===
    
    t = 0
    obj_old = 0
    obj = 1e5
    while (abs((obj - obj_old)/(obj_old+1e-6)) >= 1e-2) and (t<=30):
        obj_old = obj
        
        # cluster adjust
        objVal_adjust,objBeta,objCen,cenval,betavals,beta0val= cluReg_adjust_fips(bC,mm,cpara,lampara,wpara0,wpara1,
                                                                                  rhopara,num_clu,train_data,deltaval)
        print('objVal_adjust',objVal_adjust)
      
        # cluster assign
        obj,deltaval = cluReg_assign_fips(bC,mm,cpara,lampara,wpara0,wpara1,rhopara,ratioClu,num_clu,
                                          train_data,betavals,beta0val,cenval)
        print('obj',t,obj)
        t += 1
    
    
    # === Training Results (t-1) [not relevant to cv] ===
    
    deltaval_N = np.stack([deltaval for _ in range(batch_num)], axis=0)
    deltaval_N = deltaval_N.reshape(deltaval.shape[0]*batch_num,deltaval.shape[1])
    
    y_train_hat, clu_train_hat = ypred_clu_cty(trainN,num_clu,betavals,beta0val,deltaval_N,
                                               np.array(train_data[train_data.columns[1:]]))
    
    train_errS0, train_errS1, train_err_mix, train_count_S0, train_count_S1  = mixErr_mae_pct(trainN,mixpara,
                                                             np.array(train_data['num_drug_death_t']),
                                                             y_train_hat)
    
    print('train_err_mix',train_err_mix)
    
    # === Validation Results (t-1) ===
    
    y_val_hat, clu_val_hat = ypred_clu_cty(valN,num_clu,betavals,beta0val,deltaval,
                                           np.array(val_data[val_data.columns[1:]]))
    
    val_errS0, val_errS1, val_err_mix, val_count_S0, val_count_S1 = mixErr_mae_pct(valN,mixpara,
                                                       np.array(val_data['num_drug_death_t']),
                                                       y_val_hat)
    
    print('val_err_mix',val_err_mix)
    
    return val_err_mix, y_val_hat, clu_val_hat, betavals, beta0val,  val_errS0, val_errS1, val_count_S0, val_count_S1



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

def cross_term_names():
    name_ct = pd.read_csv('data_2_ct_names.csv')
    name_ct = np.array(name_ct).flatten()    
    name_loc = np.array(['lat','long'])
    return name_ct, name_loc

name_ct, name_loc = cross_term_names()

#----------------- Cross Validation Algorithm -----------------                    

# for each set of parameters

def CV_5splits(batch_test_year,batch_num,wpara0,wpara1,
               rhopara,num_clu,ratioClu,lampara,mixpara,cpara):
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
        
        # Step 2: call the AM algorithm to get validation error for each split
        val_err_mix, val_ypred, val_clu, val_beta, val_beta0, fpr, mae, count_S0, count_S1 = AM_Alg(train_data,val_data,batch_num,wpara0,wpara1,
                                                                      rhopara,num_clu,ratioClu,lampara,mixpara,cpara)
        val_err_splits.append(val_err_mix)
        val_fpr_splits.append(fpr)
        val_mae_splits.append(mae)
        val_count_S0_splits.append(count_S0)
        val_count_S1_splits.append(count_S1)
    
    
    
    print('val_err_splits',val_err_splits)
    return val_err_splits, val_fpr_splits, val_mae_splits, val_count_S0_splits, val_count_S1_splits

#----------------- Parameter -----------------

batch_num = 2
batch_test_year = 2021

mixpara = 100 # 10
wpara1  = 1
#cpara = [9,9,9] # for three clusters // fix to 9
ratioClu = 0.1

folds = 5

# tune: 
#rhopara = 1e2
#num_clu = 3      
#lampara = [0.05,0.05,0.05]


from sklearn.model_selection import ParameterGrid

param_grid = [
  {'num_clu': [1], 
   'rhopara': [0.1, 0.2, 0.5, 1, 5, 10], 
   'lampara': [[0.1], [0.2] ,[0.5],[1],[10]], 
   'wpara0': [0.1,1,10,100]},
  
  {'num_clu': [2], 
   'rhopara': [0.1, 0.2, 0.5, 1, 5, 10],
   'lampara': [[0.1,0.1],[0.2, 0.2], [0.5, 0.5], [1, 1], [5,5], [10,10],
               [0.1, 0.2],
               [0.1, 0.5], 
               [0.1, 1], 
               [0.1, 5], 
               [0.1, 10],
               [0.2, 0.5], 
               [0.2, 1],
               [0.2, 5],
               [0.2, 10] ,  
               [0.5, 1], 
               [0.5, 5],
               [0.5, 10],
               [1, 5], 
               [1, 10],
               [5, 10]], 
   'wpara0': [10,0.1,1,100]},
  {'num_clu': [3], 
   'rhopara': [0.1, 0.2, 0.5, 1, 5, 10], 
   'lampara': [[0.1, 0.1, 0.1], [0.2,0.2,0.2],[0.5,0.5,0.5],[1, 1, 1], [5, 5, 5],[10, 10, 10],
               [0.1, 0.2, 0.5],
               [0.1, 0.2, 1],
               [0.1, 0.2, 5],
               [0.1, 0.2, 10],
               [0.1, 0.5, 1],
               [0.1, 0.5, 5],
               [0.1, 0.5,10], 
               [0.1, 1, 5],
               [0.1, 1, 10],
               [0.1, 5, 10],
               [0.2,0.5,1],
               [0.2,0.5,5],
               [0.2,0.5,10],
               [0.2,1,5],
               [0.2,1,10],
               [0.2,5,10],
               [0.5,1,5],
               [0.5,1,10],
               [0.5,5,10],
               [1,5,10]],
               'wpara0': [10,0.1,1,100]},
  {'num_clu':[4],
   'rhopara': [0.1, 0.2, 0.5, 1, 5, 10],
   'lampara': [[0.1,0.1,0.1,0.1], [0.2, 0.2, 0.2,0.2],[0.5, 0.5, 0.5,0.5], [1, 1, 1,1], [10, 10, 10,10],[5, 5, 5,5],
               [0.1, 0.2, 0.5,1],
               [0.1, 0.2, 0.5,5],
               [0.1, 0.2, 0.5,10],
               [0.1, 0.2, 1, 5],
               [0.1, 0.2, 1, 10],
               [0.1, 0.2, 5, 10],
               [0.1, 0.5, 1, 5],
               [0.1, 0.5, 1, 10],
               [0.1, 0.5, 5, 10],
               [0.1, 1,   5, 10],
               [0.2, 0.5, 1, 5],
               [0.2, 0.5, 1, 10],
               [0.2, 0.5, 5, 10],
               [0.2, 1,   5, 10],
               [0.5, 1,   5, 10]],
               'wpara0': [10,0.1,1,100]},
  {'num_clu':[5],'rhopara': [0.1, 0.2, 0.5, 1, 5, 10],
   'lampara': [[0.1,0.1,0.1,0.1,0.1], [0.2, 0.2, 0.2,0.2,0.2],[0.5, 0.5, 0.5,0.5,0.5], [1, 1, 1,1,1], [10, 10, 10,10,10],[5,5,5,5,5],
               [0.1, 0.2, 0.5, 1, 5],
               [0.1, 0.2, 0.5, 1, 10],
               [0.1, 0.2, 0.5, 5, 10],
               [0.1, 0.2, 1,   5, 10],
               [0.1, 0.5, 1,   5, 10],
               [0.2, 0.5, 1,   5, 10]
               ],
               'wpara0': [10,0.1,1,100]}
  ]

'''
param_grid = [
  {'num_clu': [3], 'rhopara': [0.1, 1, 5, 10], 
   'lampara': [[0.01, 0.01, 0.01], [0.1, 0.1, 0.1], [0.01, 0.05, 0.1]]}
  ]

param_grid = [
  {'num_clu': [3], 'rhopara': [1,10], 
   'lampara': [[0.01, 0.01, 0.01], [0.1, 0.1, 0.1], [0.01, 0.05, 0.1]]}
  ]

param_grid = [
  {'num_clu': [5], 'rhopara': [1,10], 
   'lampara': [[0.01, 0.01, 0.01, 0.01, 0.01], [0.1, 0.1, 0.1, 0.1, 0.1], [0.01, 0.05, 0.1, 0.5, 1]]}
  ]
'''

param_list = list(ParameterGrid(param_grid))
val_err_lst = []
val_fpr_lst =[]
val_mae_lst =[]
val_count_S0_lst =[]
val_count_S1_lst =[]
for i in range(len(param_list)):
    num_clu = param_list[i]['num_clu']
    rhopara = param_list[i]['rhopara']
    lampara = param_list[i]['lampara']
    wpara0 = param_list[i]['wpara0']
    cpara = [9]*num_clu
    val_err_splits, val_fpr_splits, val_mae_splits, val_count_S0_splits, val_count_S1_splits= CV_5splits(batch_test_year,batch_num,wpara0,wpara1,
                                                       rhopara,num_clu,ratioClu,lampara,mixpara,cpara)
    
    print('iter para',i,num_clu,rhopara,lampara)

    val_err_lst.append(val_err_splits)
    val_fpr_lst.append(val_fpr_splits)
    val_mae_lst.append(val_mae_splits)
    val_count_S0_lst.append(val_count_S0_splits)
    val_count_S1_lst.append(val_count_S1_splits)


df = pd.DataFrame.from_dict(param_list)


df_err = {col: df[col].repeat(folds).reset_index(drop=True) for col in df.columns}


df_err = pd.DataFrame.from_dict(param_list)
df_err['err'] = val_err_lst
df_err['fpr'] = val_fpr_lst
df_err['mae'] = val_mae_lst
df_err['count_S0'] = val_count_S0_lst
df_err['count_S1'] = val_count_S1_lst



'''
# Step 4: choose the parameter with smallest average validation error (best one)

best_ind = avg_lst.index(min(avg_lst))
best_num_clu = param_list[best_ind]['num_clu']
best_rhopara = param_list[best_ind]['rhopara']
best_lampara = param_list[best_ind]['lampara']
print('best clu, rho, lam',best_num_clu, best_rhopara, best_lampara)

# ----------------- Testing Results (t) -----------------

# Step 5: retrain model using (t-1, t-2) and test using (t)

retrain_data, test_data = data_split_2batch(batch_test_year)
test_err_mix, test_ypred, test_clu, test_beta, test_beta0, fpr, mae = AM_Alg(retrain_data,test_data,batch_num,wpara0,wpara1,
                                                                   best_rhopara,best_num_clu,ratioClu,best_lampara,mixpara,cpara)

print('test_err_mix',test_err_mix)


# ----------------- Save Results -----------------

def test_result(test_data, test_ypred, test_clu):
    result_df = pd.DataFrame({
        'county_fips': test_data.index,  # The county fips (assumed to be the index of the DataFrame)
        'clustering_result': test_clu,
        'prediction_result': test_ypred,
        'actual_result': test_data['num_drug_death_t']
        })

    result_df['suppression_indicator'] = result_df['actual_result'].apply(lambda x: 1 if x == 0 else 0)
    
    result_df['clustering_result'] = result_df['clustering_result'] + 1
    
    result_df = result_df.reset_index(drop=True)
    result_df['new_index'] = result_df.index + 1
    
    return result_df

def test_coeff(test_data, test_beta, test_beta0):
    df = pd.DataFrame(test_beta).T
    df = df.set_index(test_data.columns[1:])
    df = pd.concat([pd.DataFrame(test_beta0).T, df])#.reset_index(drop = True)
    df = df.rename(index={0: 'beta0'})
    return df

def test_error(test_data, test_ypred, mixpara):
    N = len(test_data)
    errS0, errS1, err_mix , count_S0, count_S1 = mixErr_mae_pct(N,mixpara,
                                           np.array(test_data['num_drug_death_t']),
                                           test_ypred)
    df = pd.DataFrame(data={'errS0': [errS0], 'errS1': [errS1], 'err_mix': [err_mix]})
    return df

#df_test_result = test_result(test_data, test_ypred, test_clu)
#df_test_coeff = test_coeff(test_data, test_beta, test_beta0)
#df_test_error = test_error

'''

df_err.to_csv('grid_search_results_'+str(batch_test_year)+'_1106_v1.csv')
#df_test_result.to_csv('testing_results_'+str(batch_test_year)+'_0512_v1.csv',index=False)
#df_test_coeff.to_csv('testing_coeff_'+str(batch_test_year)+'_0512_v1.csv')
#df_test_error.to_csv('testing_error_'+str(batch_test_year)+'_0512_v1.csv')



