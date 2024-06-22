import time

import numpy as np
import numpy.matlib


def ROA_PmCalculate(VarSize, MaxIt, it):
    D = np.zeros((it))
    D[it] = (np.exp(it / (MaxIt)) - (it / MaxIt)) ** - 10
    D = D(it)
    R = np.random.rand()
    if R <= 0.5:
        SC = (np.random.rand() + np.random.randint(np.array([1, 2]))) * np.random.rand(VarSize)
        UC = (np.random.rand() + np.random.randint(np.array([1, 3]))) * np.random.rand(VarSize)
    else:
        SC = np.random.rand(VarSize)
        UC = (np.random.rand() + np.random.randint(np.array([1, 2]))) * np.random.rand(VarSize)

    return SC, UC, D



def PROPOSED(RedKites, fhd,VarMin, VarMax,MaxIt):
    D,N = RedKites.shape[0],RedKites.shape[1]
    gbest = []
    nPop = N
    nVar = D
    VarSize = np.array([1, nVar])
    costs = np.array([RedKites.Cost])
    SortOrder = np.sor(costs)
    RedKites = RedKites(SortOrder)
    ## Algorithm Defination
    BestCostArray = np.zeros((MaxIt, 1))
    BestCostArray[1] = RedKites(1).Cost
    BestAgent = RedKites[1]
    it = 0
    ct = time.time()
    for it in np.arange(it, MaxIt + 1).reshape(-1):
        ## Start Algorithm
        SC, UC, D = ROA_PmCalculate(VarSize, MaxIt, it)
        for i in np.arange(1, nPop + 1).reshape(-1):
            r=-it ((-1) / MaxIt)
            RedKites(i).Pm = D * RedKites(i).Pm + np.multiply(SC, (
                        RedKites(r).Position - BestAgent.Position)) + np.multiply(UC, (
                        BestAgent.Position - RedKites(i).Position))
            RedKites(i).NewPosition = RedKites(i).Position + RedKites(i).Pm
            RedKites(i).NewPosition = np.amax(np.amin(RedKites(i).NewPosition, VarMax), VarMin)
            RedKites(i).NewCost = fhd(np.transpose(RedKites(i).NewPosition))
            if RedKites(i).NewCost < RedKites(i).Cost:
                RedKites(i).Position = RedKites(i).NewPosition
                # if they are better than previous positions
                RedKites(i).Cost = RedKites(i).NewCost
            if RedKites(i).Cost < BestAgent.Cost:
                BestAgent = RedKites(i)
        BestCostArray[it] = BestAgent.Cost
        #     disp(['Iteration: ',num2str(it),' Best Cost = ',num2str(BestCostArray(it))]);

    ## Send Results
    BestCh = np.transpose(BestCostArray)
    gbestval = BestCh[-1]
    ct = time.time()-ct
    return gbest, gbestval, BestCh