
import pandas as pd
import numpy as np
def ypred(N,betavals,beta0val,X):
    y_hat = [0]*N
    mm = len(X[0])
    for i in range(N):
        y_hat[i] = sum(betavals[j]*X[i][j] for j in range(mm)) + beta0val
    return y_hat

def ypred_clu(N,deltaval,betavals,beta0val,X): # ypred for cluster-reg
    y_hat = [0]*N
    clu_assign = [0]*N
    mm = len(X[0])
    for i in range(N):
        l = np.where(deltaval[i]>0)
        l = l[0][0]
        clu_assign[i] = l
        y_hat[i] = sum(betavals[l][j]*X[i][j] for j in range(mm)) + beta0val[l]       
    return y_hat, clu_assign

def ypred_clu_assign_impt(N,Kclu,cenval,betavals,beta0val,X,X_impt): # ypred for cluster-reg // testing data // find centroid 
    y_hat = [0]*N
    clu_assign = [0]*N
    mm = len(X[0])
    for i in range(N):
        dist = [0]*Kclu
        for j in range(Kclu):
            dist[j] = sum((X_impt[i,:]-cenval[j,:])*(X_impt[i,:]-cenval[j,:])) # sum of squared distance
        l = dist.index(min(dist))
        clu_assign[i] = l
        y_hat[i] = sum(betavals[l][j]*X[i][j] for j in range(mm)) + beta0val[l]
    return y_hat, clu_assign

def ypred_clu_cty(N,Kclu,betavals,beta0val,bestdelta,X): # assign county based on delta
    y_hat = [0]*N
    clu_assign = [0]*N
    mm = len(X[0])
    for i in range(N):
        var = bestdelta[i]
        var = var.tolist()
        l = var.index(max(var))
        clu_assign[i] = l
        y_hat[i] = sum(betavals[l][j]*X[i][j] for j in range(mm)) + beta0val[l]
    return y_hat, clu_assign

def ypred_clu_delta(N,Kclu,deltaval,betavals,beta0val,X):
    y_hat = [0]*N
    mm = len(X[0])
    for i in range(N):
        var = deltaval[i]
        var = var.tolist()
        l = var.index(max(var))
        y_hat[i] = sum(betavals[l][j]*X[i][j] for j in range(mm)) + beta0val[l]       
    return y_hat

def mixErr_mae_pct(N,mixpara,y,y_hat):
    censor_max = 10
    z_hat = [0]*N
    MAE_set = []
    MSE_set = []
    MAPE_set = []
    count_S0 = 0
    count_S1 = 0
    for i in range(N):
        if y[i]==0: #supp
            count_S0 += 1
            # binary error
            if y_hat[i] >= censor_max:
                z_hat[i] = 1 # wrong pred
        else:
            count_S1 += 1
            # mean absolute error // pct
            if y_hat[i]<0:
                #print('neg',y_hat[i])
                y_hat[i] = 0
            MAPE_val = 100*abs((y_hat[i]-y[i])/y[i])
            #print('mape, yhat, y',MAPE_val,y_hat[i],y[i])
            MAPE_set.append(MAPE_val)
            
            MAE_val = abs(y_hat[i]-y[i])
            MAE_set.append(MAE_val)
            MSE_val = (y_hat[i]-y[i])**2
            MSE_set.append(MSE_val)
    
    err_S0 = sum(z_hat[i] for i in range(N)) # num wrong pred supp
    err_S0_rate = err_S0/count_S0
    
    err_S1 = sum(MAE_set)/count_S1
    err_S1_mse = sum(MSE_set)/count_S1
    err_S1_pct = sum(MAPE_set)/count_S1

    err_mix = mixpara * (count_S0/N) * err_S0_rate + (count_S1/N) * err_S1
    return err_S0_rate, err_S1, err_mix, count_S0,count_S1 #err_S1_pct, err_S1_mse
