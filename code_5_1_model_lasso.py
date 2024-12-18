
import pandas as pd
import numpy as np
from gurobipy import *

def Model_lasso_mae(X_train, y_train, lampara):
    mb = Model()
    N = len(y_train)
    mm = len(X_train[0])
    
    betavars = mb.addVars(mm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')
    beta0var = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta0')   
    auxbetas = mb.addVars(mm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxbeta')
    auxbeta0 = mb.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='auxbeta0')
    aux3 = mb.addVars(N, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux3")

    mb.setObjective(1/float(N)*quicksum(aux3[i] for i in range(N))
                    +lampara*(quicksum(auxbetas[j] for j in range(mm))+auxbeta0),GRB.MINIMIZE)
    mb.update()

    mb.addConstrs((betavars[j] <= auxbetas[j] for j in range(mm)), name="aux_betas")
    mb.addConstrs((-betavars[j] <= auxbetas[j] for j in range(mm)), name="aux_betas")
    mb.addConstr((beta0var <= auxbeta0), name="aux_beta0")
    mb.addConstr((-beta0var <= auxbeta0), name="aux_beta0")
    mb.update()
    
    mb.addConstrs((aux3[i] >= (quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-y_train[i]) for i in range(N)), name="aux_constr3")
    mb.addConstrs((aux3[i] >= -(quicksum(betavars[j]*X_train[i][j] for j in range(mm))+beta0var-y_train[i]) for i in range(N)), name="aux_constr3")
    mb.update()
    
    mb.params.OutputFlag = 0
    #mb.params.timelimit = 300 
    mb.optimize()
    
    betavals= [betavars[i].x for i in range(mm)]
    beta0val= beta0var.x

    return betavals, beta0val


