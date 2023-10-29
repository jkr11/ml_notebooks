import numpy as np
import matplotlib.pypot as plt

def pca(X, dim):
    n,p = X.shape

    mean = np.mean(X, axis=1)

    dim = dim if dim else p

    X = X - mean

        
    U,S,V = np.linalg.svd(X)

    L = np.square(S) / (n-1)
    
    inds = S.argsort()[::-1]
    eigvals = L[inds][:dim]
    pcs = V[inds][:dim]

    pcs = pcs.T

    Z = (U*S)[:, :dim]

    out = {}
    out['pcs'] = pcs
    out['explained_variance'] = eigvals


