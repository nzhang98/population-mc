from scipy import stats
import numpy as np
from collections import Counter
import seaborn as sns
import pandas as pd

def generate_dataset(n, mu_1 = 0, mu_2 = 2, sig_1 = 1, sig_2 = 1, alpha = 0.2, set_seed = None, form= 'mix-norm'):
    X = np.zeros(n)

    assert (0 <= alpha <= 1), "alpha proportion outside of [0,1] range"

    if isinstance(set_seed, int):
        np.random.seed(set_seed)

    if form == 'mix-norm':
        X = np.random.normal(mu_1, sig_1, n)*alpha + np.random.normal(mu_2, sig_2, n)*(1-alpha)
        return X
    else:
        print('Distributional form not correctly specified')
        return

def generate_mixGauss(n, mu_1 = 0, mu_2 = 2, sig_1 = 1, sig_2 = 1, alpha = 0.2, set_seed = None, form= 'mix-norm'):
    X = np.zeros(n)
    assert (0 <= alpha <= 1), "alpha proportion outside of [0,1] range"

    if isinstance(set_seed, int):
        np.random.seed(set_seed)
    
    if form == 'mix-norm':
        for i in range(n):
            draw = np.random.uniform()
            if draw < alpha:
                X[i] = np.random.normal(mu_1, sig_1)
            else:
                X[i] = np.random.normal(mu_2, sig_2)

    return X

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
        # mu0[:,0] = 0
        mu0[:, -1] = [i for sublist in [[x]*int(N/p) for x in V] for i in sublist]
        mu0_base = np.full((baseline, n_parameters+1), np.mean(X))
        # mu0_base[:,1] = 2
        mu0_base[:, -1] = [i for sublist in [[x]*int(baseline/p) for x in V] for i in sublist]

        mu0 = np.concatenate([mu0, mu0_base])

    # mu0[:,0] = -2
    # mu0[:,1] = 5
    mu = mu0.copy()
    mus = [mu0]
    

    W = np.zeros(N + baseline)
    for t in range(T):
        if t%50 == 0:
            print(t)
        mu_star = mu.copy()
        for n in range(N+baseline):
            for i in range(len(mu[n]) - 1):
                mu_star[n][i] = np.random.normal(mu[n][i], np.sqrt(mu[n][-1]))

            W[n] = np.sum(np.log(GaussianLlhood(X, mu_star[n][0], sigma = np.sqrt(1))*alpha + GaussianLlhood(X, mu_star[n][1], sigma = np.sqrt(1))*(1-alpha))) +\
            np.log(GaussianPrior(mu_star[n][0])*GaussianPrior(mu_star[n][1])) -\
            (np.log(GaussianLlhood(mu_star[n][0], mu = mu[n][0], sigma = np.sqrt(mu[n][-1]))) + \
            np.log(GaussianLlhood(mu_star[n][1], mu = mu[n][1], sigma = np.sqrt(mu[n][-1]))))
        
        #Resample
        # W /= sum(W)
        # print('t: ', t, 'W: ', W)
        print('t: ', t)

        P = np.exp(W-np.max(W))
        weights_normalized = P/np.sum(P)   

        resample_idx = np.random.choice(N+baseline, N+baseline, p  = weights_normalized)
        mu = np.array([mu_star[x] for x in resample_idx])

        mu[-baseline:, -1] = [i for sublist in [[x]*int(baseline/p) for x in V] for i in sublist] #TO UPDATE IN FINAL VERSION CAREFUL!!!!
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


X = generate_dataset(1000, mu_1 = 0, mu_2 = 2, sig_1 = np.sqrt(1), sig_2 = np.sqrt(1), alpha = 0.5, set_seed = 2)
X = generate_mixGauss(1000, mu_1 = 0, mu_2 = 2, sig_1 = np.sqrt(1), sig_2 = np.sqrt(1), alpha = 0.2, set_seed = 2)

# test = np.random.multivariate_normal((0,2), np.eye(2), (1000))
# X = test[:,0]*0.2 + test[:,1]*0.8

a = np.random.normal(0, 1, 200)
b = np.random.normal(2, 1, 800)
X = np.concatenate((a,b))

sns.distplot(X)
V = [5, 2, 0.1, 0.05, 0.01]

mus, R = GaussianMixturePMC(X, V, 1000, baseline_threshold=0.01, alpha = 0.2)

R = pd.DataFrame(R)

import pickle
with open('mus.pkl', 'wb') as f:
    pickle.dump(mus, f)

with open('Rs.pkl', 'wb') as f:
    pickle.dump(R, f)

with open('Rs.pkl', 'rb') as f:
   test = pickle.load(f)

N = len(X)
p = len(V)
baseline_threshold = 0.01
baseline = int(N*baseline_threshold*p)
n_parameters = 2
alpha = 0.2

# np.exp(np.sum(
for n in range(N+baseline):
    # for i in range(len(mu[n]) - 1):
    #     mu_star[n][i] = np.random.normal(mu[n][i], np.sqrt(mu[n][-1]))

    W[n] = np.sum(np.log(GaussianLlhood(X[n], mu_star[n][0])*alpha + GaussianLlhood(X[n], mu_star[n][1])*(1-alpha))) +\
    np.log(GaussianPrior(mu_star[n][0])*GaussianPrior(mu_star[n][1])) -\
    (np.log(GaussianLlhood(mu_star[n][0], mu = mu[n][0], sigma = np.sqrt(mu[n][-1]))) + \
    np.log(GaussianLlhood(mu_star[n][1], mu = mu[n][1], sigma = np.sqrt(mu[n][-1]))))

P = W-max(W)
weights_normalized = P/sum(P)


np.sum(np.log(GaussianLlhood(X, 0)*alpha + GaussianLlhood(X, 1.6)*(1-alpha)))# +\
np.log(GaussianPrior(mu_star[n][0])*GaussianPrior(mu_star[n][1])) -\
(np.log(GaussianLlhood(mu_star[n][0], mu = mu[n][0], sigma = np.sqrt(mu[n][-1]))) + \
np.log(GaussianLlhood(mu_star[n][1], mu = mu[n][1], sigma = np.sqrt(mu[n][-1]))))

def computeLlhood(x, mu1, mu2, sigma1, sigma2, alpha):
    return GaussianLlhood(x, mu = mu1, sigma = sigma1)*alpha + GaussianLlhood(x, mu= mu2, sigma =sigma2)*(1-alpha)

np.sum(np.log(computeLlhood(X, 0, 2, np.sqrt(1), np.sqrt(1), 0.2)))
np.sum(np.log(computeLlhood(X, 1.6, 1.6, np.sqrt(1), np.sqrt(1), 0.2)))
np.log(GaussianPrior(1.6)*GaussianPrior(1.6))


np.prod(computeLlhood(X, 0, 2, np.sqrt(1), np.sqrt(0.1), 0.2))

np.mean(stats.norm(0, 1).pdf(X))*0.2
np.mean(stats.norm(2, 1).pdf(X))*0.8

np.mean(stats.norm(1.6, 1).pdf(X))


np.sum(np.log(GaussianLlhood(X, mu = 1.85, sigma = np.sqrt(5))*alpha + GaussianLlhood(X, mu = 1.85, sigma = np.sqrt(10))*(1-alpha)))
np.sum(np.log(GaussianLlhood(X, mu = 0, sigma = np.sqrt(5))*alpha + GaussianLlhood(X, mu = 2, sigma = np.sqrt(10))*(1-alpha)))
sns.lineplot(np.cumsum(np.mean(mus,1),0)[:,0]/np.arange(len(mus)))
sns.lineplot(np.cumsum(np.mean(mus,1),0)[:,1]/np.arange(len(mus)))


sns.lineplot(np.mean(mus, 1)[:,0])
sns.lineplot(np.mean(mus, 1)[:,1])

np.mean(np.mean(mus,1)[10:,0])
np.mean(np.mean(mus,1)[10:,1])

R = pd.DataFrame(R)
sns.lineplot(R.iloc[:,:2])
sns.lineplot(R.iloc[:,2:])

sns.distplot(X)

for i in range(-10,20):
    for j in range(10,30):
        print((i/10, j/10), np.sum(np.log(GaussianLlhood(X,i/10)*alpha + GaussianLlhood(X, j/10)*(1-alpha))))
        

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