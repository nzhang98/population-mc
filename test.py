from scipy import stats
import numpy as np
from collections import Counter

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

def GaussianMixturePMC(X, V, T, n_parameters = 2, cold_start = True, baseline_threshold = 0.01, alpha = 0.5):
    N = len(X)
    p = len(V)

    baseline = int(N*baseline_threshold*p) #Used to avoid certain variances from vanishing

    assert (N%p == 0), "N%p is not divisible" 
    R = [Counter(V)]

    if cold_start:
        mu0 = np.full((N, n_parameters+1), np.mean(X))
        mu0[:, -1] = [i for sublist in [[x]*p for x in V] for i in sublist]
        mu0_base = np.full((baseline, n_parameters+1), np.mean(X))
        mu0_base[:, -1] = [i for i in V]

        mu0 = np.concatenate([mu0, mu0_base])

    mu = mu0.copy()
    mus = [mu0]
    

    W = np.zeros(N + baseline)
    for t in range(T):
        mu_star = mu.copy()
        for n in range(N+baseline):
            for i in range(len(mu[n]) - 1):
                mu_star[n][i] = np.random.normal(mu[n][i], np.sqrt(mu[n][-1]))
            

            W[n] = np.exp(np.sum(np.log((GaussianLlhood(X, mu_star[n][0])*alpha + GaussianLlhood(X, mu_star[n][1])*(1-alpha))))) *\
                    GaussianPrior(mu_star[n][0])*GaussianPrior(mu_star[n][1]) /\
                    (GaussianLlhood(mu_star[n][0], mu = mu[n][0], sigma = np.sqrt(mu[n][-1])) * \
                    GaussianLlhood(mu_star[n][1], mu = mu[n][1], sigma = np.sqrt(mu[n][-1])))

        
        #Resample
        W /= sum(W)
        resample_idx = np.random.choice(N+baseline, N+baseline, p  = W)
        mu = np.array([mu_star[x] for x in resample_idx])

        mu[-baseline:, -1] = V #TO UPDATE IN FINAL VERSION CAREFUL!!!!
        V_assignments = mu[:, -1].copy()
        R.append(Counter(V_assignments))
        mus.append(mu.copy())

        #Reshuffle Variances
        np.random.shuffle(V_assignments)
        mu[:, -1] = V_assignments

    return mus, R



resample_idx = np.random.choice(len(mus), 5, p  = test)
mus = [mus[x] for x in resample_idx]

np.set_printoptions(suppress=True)


X = generate_dataset(9, mu_1 = 1, mu_2 = 2, sig_1 = 1, sig_2 = 1, alpha = 0.2, set_seed = 42)
V = [0.1, 1, 5]

mus, R = GaussianMixturePMC(X, V, 10, baseline_threshold=0.11112, alpha = 0.2)



N = len(X)
p = len(V)
baseline_threshold = 0.01
baseline = N*baseline_threshold*p
n_parameters = 2



assert (N%p == 0), "N%p is not divisible" 
R = [N//p for _ in V]

mu0 = np.full((N, n_parameters+1), np.mean(X))
mu0[:, -1] = [i for sublist in [[x]*p for x in V] for i in sublist]
mu0_base = np.full((3, n_parameters+1), np.mean(X))
mu0_base[:, -1] = [i for i in V]

mu = np.concatenate([mu0, mu0_base])

test = X[0]

mu_1 = np.mean(X)
mu_2 = np.mean(X)

mu_1_star = np.random.normal(mu_1, 1)
mu_2_star = np.random.normal(mu_2, 1)

stats.norm(mu_1, 1).pdf(mu_1_star)

stats.norm(mu_1_star, 1).pdf(test)*alpha + \
stats.norm(mu_2_star, 1).pdf(test)*(1-alpha)


#RESAMPLING
test = np.arange(5)
test = test/sum(test)


mus = [list(x) for x in np.arange(10).reshape(5,2)]
for x in range(len(mus)):
    mus[x] = [mus[x], x]

resample_idx = np.random.choice(len(mus), 5, p  = test)
mus = [mus[x] for x in resample_idx]

for i in range(len(resample_idx)):
    r_k = 