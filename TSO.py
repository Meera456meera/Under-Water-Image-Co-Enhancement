import time
from math import inf

import numpy as np


def TSO(T, Max_iter, Low, Up, fobj):
    N,Dim = T.shape[0],T.shape[1]
    Tuna1 = np.zeros((1, Dim))
    Tuna1_fit = inf

    Iter = 0
    aa = 0.7
    z = 0.05
    Convergence_curve = []
    ct = time.time()
    while Iter < Max_iter:

        C = Iter / Max_iter
        a1 = aa + (1 - aa) * C
        a2 = (1 - aa) - (1 - aa) * C
        fitness = []
        for i in np.arange(1, T.shape[1 - 1] + 1).reshape(-1):
            Flag4ub = T[i,:]> Up
            Flag4lb = T[i,:] < Low
            T[i, :] = (np.multiply(T[i,:], (not (Flag4ub + Flag4lb)))) + np.multiply(Up, Flag4ub) + np.multiply(Low,
                                                                                                                Flag4lb)
            fitness[i] = fobj(T[i,:])
            if fitness[i] < Tuna1_fit:
                Tuna1_fit = fitness[i]
                Tuna1 = T[i,:]
            # ---------------- Memory saving-------------------
        if Iter == 0:
            fit_old = fitness
            C_old = T
        for i in np.arange(1, T + 1).reshape(-1):
            if fit_old[i] < fitness[i]:
                fitness[i] = fit_old[i]
                T[i, :] = C_old[i,:]
        C_old = T
        fit_old = fitness
        # -------------------------------------------------
        t = (1 - Iter / Max_iter) ** (Iter / Max_iter)
        if np.random.rand() < z:
            T[1, :] = (Up - Low) * np.random.rand() + Low
        else:
            if 0.5 < np.random.rand():
                r1 = np.random.rand()
                Beta = np.exp(r1 * np.exp(3 * np.cos(np.pi * ((Max_iter - Iter + 1) / Max_iter)))) * (
                    np.cos(2 * np.pi * r1))
                if C > np.random.rand():
                    T[1, :] = np.multiply(a1, (Tuna1 + Beta * np.abs(Tuna1 - T[1,:]))) + np.multiply(a2,
                                                                                                     T[1,:])
                else:
                    IndivRand = np.multiply(np.random.rand(1, Dim), (Up - Low)) + Low
                    T[1, :] = np.multiply(a1, (IndivRand + Beta * np.abs(IndivRand - T[i,
                                       :]))) + np.multiply(a2, T[1,:])
            else:
                TF = (np.random.rand() > 0.5) * 2 - 1
                if 0.5 > np.random.rand():
                    T[1, :] = Tuna1 + np.multiply(np.random.rand(1, Dim),
                                                  (Tuna1 - T[1,:])) + np.multiply(TF, t ** 2.0) * (Tuna1 - T[1,:])
                else:
                    T[1, :] = np.multiply(TF, t ** 2.0) * T[1,:]
        for i in np.arange(2, T + 1).reshape(-1):
            r = np.random.rand()
            if np.random.rand() < z:
                T[i, :] = (Up - Low) * r + Low
            else:
                if 0.5 < np.random.rand():
                    r1 = np.random.rand()
                    Beta = np.exp(r1 * np.exp(
                        3 * np.cos(np.pi * ((Max_iter - Iter + 1) / Max_iter)))) * (
                               np.cos(2 * np.pi * r1))
                    if C > np.random.rand():
                        T[i, :] = np.multiply(a1, (Tuna1 + Beta * np.abs(Tuna1 - T[i,
                                                   :]))) + np.multiply(a2, T[i - 1,:])
                    else:
                        IndivRand = np.multiply(np.random.rand(1, Dim), (Up - Low)) + Low
                        T[i, :] = np.multiply(a1,
                                          (IndivRand + Beta * np.abs(IndivRand - T[i,
                                           :]))) + np.multiply(a2, T[i - 1,:])
                else:
                    TF = ( np.random.rand() > 0.5) * 2 - 1
                    if 0.5 >  np.random.rand():
                        T[i, :] = Tuna1 + np.multiply(np.random.rand(1, Dim),
                                                      (Tuna1 - T[i,
                                                       :])) + TF * t ** 2.0 * (
                                    Tuna1 - T[i,:])
                    else:
                      T[i, :] = TF * t ** 2.0 * T[i,:]
    Iter = Iter + 1
    Convergence_curve[Iter] = Tuna1_fit
    ct = time.time() - ct

    return Tuna1_fit, Tuna1, Convergence_curve, ct