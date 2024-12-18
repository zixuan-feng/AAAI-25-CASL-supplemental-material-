
import pandas as pd
import numpy as np
from gurobipy import *

def Model2_L1(bC,mm,cpara,lampara,wpara0,wpara1,X_train,y_train,N0,N1,K):
    mb = Model()
    bC = bC
    N = len(y_train)
    
    S0 = [] #suppressed
    S1 = [] #non-suppressed
    for i in range(N):
        if y_train[i]==0:
            S0.append(i)
        else:
            S1.append(i)
    
    #N0 = len(S0)
    #N1 = len(S1)
    #print('model2 N0,N1',N0,N1)
    betavars = mb.addVars(mm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')
    beta0var = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta0')   
    auxbetas = mb.addVars(mm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxbeta')
    auxbeta0 = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxbeta0')
        
    aux1 = mb.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux1")
    aux2 = mb.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux2")
    aux3 = mb.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux3")

    
    mb.setObjective(wpara0/float(N0)*quicksum(aux2[i] for i in S0) 
                    +wpara1/float(N1)*quicksum(aux3[i] for i in S1)
                    +lampara/float(K)*(quicksum(auxbetas[j] for j in range(mm))+auxbeta0),GRB.MINIMIZE)
    mb.update()

    mb.addConstrs((betavars[j] <= auxbetas[j] for j in range(mm)), name="aux_betas")
    mb.addConstrs((-betavars[j] <= auxbetas[j] for j in range(mm)), name="aux_betas")
    mb.addConstr((beta0var <= auxbeta0), name="aux_beta0")
    mb.addConstr((-beta0var <= auxbeta0), name="aux_beta0")
    mb.update()

    mb.addConstrs((quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-cpara == aux1[i] for i in S0), name="aux_constr1")
    mb.addConstrs((aux2[i] == max_([aux1[i],0.0]) for i in S0), name="aux_constr2")    
    mb.update()
    
    mb.addConstrs((aux3[i] >= (quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-y_train[i]) for i in S1), name="aux_constr3")
    mb.addConstrs((aux3[i] >= -(quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-y_train[i]) for i in S1), name="aux_constr3")
    mb.update()
    
    mb.params.OutputFlag = 0
    mb.params.timelimit = 300 
    mb.optimize()
    
    betavals= mb.getAttr('x', betavars)
    beta0val= beta0var.x
    aux1= mb.getAttr('x', aux1)
    aux2= mb.getAttr('x', aux2)
    aux3= mb.getAttr('x', aux3)
    auxbetas= mb.getAttr('x', auxbetas)
    auxbeta0= auxbeta0.x
    
    return mb.objVal,betavals,beta0val #,aux1,aux2,aux3,N0,N1,auxbetas,auxbeta0


def Model2_L1_ex_hinge_loss(bC,mm,cpara,lampara,wpara0,wpara1,X_train,y_train,N0,N1,K):
    mb = Model()
    bC = bC
    N = len(y_train)
    
    S0 = [] #suppressed
    S1 = [] #non-suppressed
    for i in range(N):
        S1.append(i)
    
    #N0 = len(S0)
    #N1 = len(S1)
    #print('model2 N0,N1',N0,N1)
    betavars = mb.addVars(mm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')
    beta0var = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta0')   
    auxbetas = mb.addVars(mm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxbeta')
    auxbeta0 = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxbeta0')
        
    aux1 = mb.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux1")
    aux2 = mb.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux2")
    aux3 = mb.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux3")

    
    mb.setObjective(wpara0/float(N0)*quicksum(aux2[i] for i in S0) 
                    +wpara1/float(N1)*quicksum(aux3[i] for i in S1)
                    +lampara/float(K)*(quicksum(auxbetas[j] for j in range(mm))+auxbeta0),GRB.MINIMIZE)
    mb.update()

    mb.addConstrs((betavars[j] <= auxbetas[j] for j in range(mm)), name="aux_betas")
    mb.addConstrs((-betavars[j] <= auxbetas[j] for j in range(mm)), name="aux_betas")
    mb.addConstr((beta0var <= auxbeta0), name="aux_beta0")
    mb.addConstr((-beta0var <= auxbeta0), name="aux_beta0")
    mb.update()

    mb.addConstrs((quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-cpara == aux1[i] for i in S0), name="aux_constr1")
    mb.addConstrs((aux2[i] == max_([aux1[i],0.0]) for i in S0), name="aux_constr2")    
    mb.update()
    
    mb.addConstrs((aux3[i] >= (quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-y_train[i]) for i in S1), name="aux_constr3")
    mb.addConstrs((aux3[i] >= -(quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-y_train[i]) for i in S1), name="aux_constr3")
    mb.update()
    
    mb.params.OutputFlag = 0
    mb.params.timelimit = 300 
    mb.optimize()
    
    betavals= mb.getAttr('x', betavars)
    beta0val= beta0var.x
    aux1= mb.getAttr('x', aux1)
    aux2= mb.getAttr('x', aux2)
    aux3= mb.getAttr('x', aux3)
    auxbetas= mb.getAttr('x', auxbetas)
    auxbeta0= auxbeta0.x
    
    return mb.objVal,betavals,beta0val #,aux1,aux2,aux3,N0,N1,auxbetas,auxbeta0
