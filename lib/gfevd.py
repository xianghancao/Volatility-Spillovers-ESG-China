from .base import * 
from tqdm.notebook import tqdm
import numpy as np


def compute_gfevd(H, A, sigma_u):
    n_vars = A.shape[1]
    gfevd = np.zeros((n_vars, n_vars))
    e = np.eye(n_vars)

    A_sigma_u = np.array([A[h] @ sigma_u for h in range(H)])
    eA_sigma_u = np.array([[e[i].T @ A_sigma_u[h] for h in range(H)] for i in range(n_vars)])
    den_cache = np.array([np.sum([eA_sigma_u[i, h] @ A[h].T @ e[i] for h in range(H)]) for i in range(n_vars)])

    for i in (range(n_vars)):
        for j in range(n_vars):
            num = 1. / sigma_u[j, j] * np.sum([(eA_sigma_u[i, h] @ e[j]) ** 2 for h in range(H)])
            gfevd[i, j] = num / den_cache[i]

    return gfevd

def compute_gfevd_ori(H, A, sigma_u ):
    n_vars = A.shape[1]
    gfevd = np.zeros((n_vars, n_vars))
    e = np.eye(n_vars)

    for i in tqdm(range(n_vars)):
        for j in range(n_vars):
            if sigma_u[j,j] == 0:
                gfevd[i,j] = 0
                continue
            num = 1. / sigma_u[j, j] * np.sum([(e[i].T @ A[h] @ sigma_u @ e[j]) ** 2 for h in range(H)]) 
            den = np.sum([(e[i].T @ A[h] @ sigma_u @ A[h].T @ e[i]) for h in range(H)])
            gfevd[i, j] = num / den

    return gfevd
    
    
