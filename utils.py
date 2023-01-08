import numpy as np
import scipy as sp

def generate_dataset(n, mu_1 = 0, mu_2 = 2, sig_1 = 1, sig_2 = 1, alpha = 0.5, set_seed = None, form= 'mix-norm'):
    X = np.zeros(n)

    assert (0 <= alpha <= 1), "alpha proportion outside of [0,1] range"

    if isinstance(set_seed, int):
        np.random.seed(set_seed)

    if form == 'mix-norm':
        for i in range(n):
            X[i] = np.random.normal(mu_1, sig_1)*alpha + np.random.normal(mu_2, sig_2)*(1-alpha)
        return X
    else:
        print('Distributional form not correctly specified')
        return

def GaussianPrior(x, loc = 1, sigma = 1, lambd = 0.1):
    return stats.norm(loc, np.sqrt((sigma**2)/lambd)).pdf(x)

def GaussianLlhood(x, mu, sigma = 1):
    return stats.norm(mu, sigma).pdf(x)