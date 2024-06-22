import numpy as np

from Glob_Vars import Glob_Vars
from Model_Transunet import Model_TransUnet


def Objective_Seg(Soln):
    images = Glob_Vars.Images
    GT = Glob_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        per = round(images.shape[0] * 0.75)
        train_data = images[:per]
        train_target = GT[:per]
        test_data = images[per:]
        test_target = GT[per:]
        Eval, Images = Model_TransUnet(train_data, train_data, test_data, test_target, sol.astype('int'))
        Fitn[i] =(1/ Eval[1] )
    return Fitn



