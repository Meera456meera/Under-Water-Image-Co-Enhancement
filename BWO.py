import time
from math import inf
import numpy as np
import scipy
from scipy import rand


def BWO(pos, fobj, lb, ub,Max_it):
    # disp('Beluga Whale Optimization is optimizing your problem');
    Npop,nD = pos.shape[0],pos.shape[1]
    fit = inf * np.ones((Npop, 1))
    newfit = fit
    Curve = inf * np.ones((1, Max_it))
    kk = np.zeros((1, Max_it))
    Counts_run = 0
    if ub.shape[2 - 1] == 1:
        lb = lb * np.ones((1, nD))
        ub = ub * np.ones((1, nD))
    pos = np.multiply(np.random.rand(Npop, nD), (ub - lb)) + lb
    for i in np.arange(1, Npop + 1).reshape(-1):
        fit[i, 1] = fobj(pos[i, :])
        Counts_run = Counts_run + 1
    fvalbest, index = np.amin(fit)
    xposbest = pos[index, :]
    kk_Record = []
    T = 0
    ct = time.time()
    while T <= Max_it:
        newpos = pos
        WF = 0.1 - 0.05 * (T / Max_it)
        kk = (1 - 0.5 * T / Max_it) * np.random.rand(Npop, 1)
        for i in np.arange(1, Npop + 1).reshape(-1):
            if kk(i) > 0.5:
                r1 = np.random.rand()
                r2 = np.random.rand()
                RJ = np.ceil(Npop * rand)
                while RJ == i:
                    RJ = np.ceil(Npop * rand)

                if nD <= Npop / 5:
                    params = randperm(nD, 2)
                    newpos[i, params[1]] = pos[i, params[0]] + [pos[RJ, params[0]] - pos[i, params[1]]] * (
                                r1 + 1) * np.sin(r2 * 360)
                    newpos[i, params[2]] = pos[i, params[1]] + [pos[RJ, params[0]] - pos[i, params[1]]] * (
                                r1 + 1) * np.cos(r2 * 360)
                else:
                    params = randperm(nD)
                    for j in np.arange(1, int(np.floor(nD / 2)) + 1).reshape(-1):
                        newpos[i, 2 * j - 1] = pos[i, params[2 * j - 1]] + [
                            pos[RJ, params[1]] - pos[i, params[2 * j - 1]]] * (r1 + 1) * np.sin(r2 * 360)
                        newpos[i, 2 * j] = pos[i, params[2 * j]] + [pos[RJ, params[1]] - pos[i, params[2 * j]]] * (
                                    r1 + 1) * np.cos(r2 * 360)
            else:
                r3 = np.random.rand()
                r4 = np.random.rand()
                C1 = 2 * r4 * (1 - T / Max_it)
                RJ = np.ceil(Npop * rand)
                while RJ == i:
                    RJ = np.ceil(Npop * rand)

                alpha = 3 / 2
                sigma = (scipy.special.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / (
                            scipy.special.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
                u = np.multiply(np.random.randn(1, nD), sigma)
                v = np.random.randn(1, nD)
                S = u / np.abs(v) ** (1 / alpha)
                KD = 0.05
                LevyFlight = np.multiply(KD, S)
                newpos[i, :] = r3 * xposbest - r4 * pos[i, :] + np.multiply(C1 * LevyFlight, (pos[RJ, :] - pos[i, :]))
            # boundary
            Flag4ub = newpos[i, :] > ub
            Flag4lb = newpos[i, :] < lb
            newpos[i, :] = (np.multiply(newpos[i, :], (not (Flag4ub + Flag4lb)))) + np.multiply(ub,
                                                                                                Flag4ub) + np.multiply(
                lb, Flag4lb)
            newfit[i, 1] = fobj(newpos[i, :])
            Counts_run = Counts_run + 1
            if newfit(i, 1) < fit(i, 1):
                pos[i, :] = newpos[i, :]
                fit[i, 1] = newfit[i, 1]
        for i in np.arange(1, Npop + 1).reshape(-1):
            # whale falls
            if kk(i) <= WF:
                RJ = np.ceil(Npop * rand)
                r5 = np.random.rand()
                r6 = np.random.rand()
                r7 = np.random.rand()
                C2 = 2 * Npop * WF
                stepsize2 = r7 * (ub - lb) * np.exp(- C2 * T / Max_it)
                newpos[i, :] = (r5 * pos[i, :] - r6 * pos[RJ, :]) + stepsize2
                # boundary
                Flag4ub = newpos[i, :] > ub
                Flag4lb = newpos[i, :] < lb
                newpos[i, :] = (np.multiply(newpos[i, :], (not (Flag4ub + Flag4lb)))) + np.multiply(ub,
                                                                                                    Flag4ub) + np.multiply(
                    lb, Flag4lb)
                newfit[i, 1] = fobj(newpos[i, :])
                Counts_run = Counts_run + 1
                if newfit(i, 1) < fit(i, 1):
                    pos[i, :] = newpos[i, :]
                    fit[i, 1] = newfit(i, 1)
        fval, index = np.amin(fit)
        if fval < fvalbest:
            fvalbest = fval
            xposbest = pos[index, :]
        kk_Record[T] = kk(1)
        Curve[T] = fvalbest
        T = T + 1
    ct = time.time()-ct
    # display(['The function call is ', num2str(Counts_run)]);
    return xposbest, fvalbest, Curve,ct