# Run before importing numpy to force each process running single-threaded
# Use numpy.__config__.show() to see if you are using OpenBLAS or MKL
# OPENBLAS_NUM_THREADS=1 or export MKL_NUM_THREADS=1
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
from numba import njit
from numpy.linalg import norm
from sklearn.linear_model import Lasso
from celer.datasets import make_correlated_data
from utils import ST, lasso_loss
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import cpu_count

# have more readable plots with increased fontsize:
fontsize = 16
plt.rcParams.update({'axes.labelsize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize - 2,
              'ytick.labelsize': fontsize - 2})
n, m, s = 100, 2000, 0.01
n_proc = 10
n_exp = 10
acc = 5e-2
cond_n = -11
mu = 0
noise_scale = 0.01

A, y, x_true = make_correlated_data(n, m, density=s, random_state=0)
lbda_max = norm(A.T @ y, ord=np.inf)
lbda = (lbda_max / 100)

l_res = np.max(norm(A.T@A, axis=0))

clf = Lasso(alpha=lbda/len(y), fit_intercept=False,
            max_iter=200000, tol=1e-10)

clf.fit(A, y)
f_star = lasso_loss(A, y, lbda, clf.coef_)
list_time = np.zeros(n_proc)
list_time_all = np.zeros((n_exp,n_proc))

@njit
def asyn_prox_cd(A, y, lbda, x, fx, gammas, f_star):
    while fx[0] > acc:
        i = np.random.randint(m)
        neg_grad_i = A[:, i] @ (y - A @ x )
        new = ST(x[i] + neg_grad_i * gammas[i], lbda * gammas[i])
        x[i] = new
        f = lasso_loss(A, y, lbda, x)
        fx[0] = f - f_star
        
if __name__ == '__main__':

    for exp in range(n_exp):
        for tau in [1,10]:
            if tau == 1 and not (exp == 0):
                continue
            for j in range(n_proc):
                if tau == 1 and not (j == 0):
                    continue
                if tau == 1 and j == 0:
                    j = int(cpu_count() / 4)
                np.random.seed(exp)
                N_PROC = j+1
                x = np.zeros(m)
                step = .5*( norm(A, axis=0)**2 + (2*(tau)*l_res)/np.sqrt(m))
                gammas = 1. / (step)
                fx = 3 * np.ones(1)

                with SharedMemoryManager() as smm:
                    shm_x = smm.SharedMemory(size=x.nbytes)
                    # Now create a NumPy array backed by shared memory
                    x_shared = np.ndarray(x.shape, dtype=x.dtype, buffer=shm_x.buf)
                    x_shared[:] = x[:]

                    shm_fx = smm.SharedMemory(size=fx.nbytes)
                    fx_shared = np.ndarray(fx.shape, dtype=fx.dtype, buffer=shm_fx.buf)
                    fx_shared[:] = fx[:]

                    processes = [mp.Process(target=asyn_prox_cd, args=(A, y, lbda, x_shared, fx_shared, gammas, f_star)) 
                                 for _ in range(N_PROC)]

                    start = time.perf_counter()
                    for proc in processes:
                        proc.start()

                    for proc in processes:
                        proc.join()

                    stop = time.perf_counter()

                    for proc in processes:
                        proc.terminate()

                    if not (tau == 1):
                        list_time[j] = stop - start

                shm_x.close()   # Close each SharedMemory instance
                shm_fx.close()   # Close each SharedMemory instance
        list_time_all[exp] =  (1 / list_time) * list_time[0]

    mean = np.mean(list_time_all, axis=0)
    std = np.std(list_time_all, axis=0)
    plt.figure(constrained_layout=True)
    plt.fill_between(np.array(range(n_proc))+1, mean + std, mean - std, alpha=.3)
    plt.plot(np.array(range(n_proc))+1, mean, label=r"Algo 1.1", linewidth=3
            , marker='o', fillstyle='none', linestyle=(0, (5, 5)))
    plt.plot(np.array(range(n_proc))+1, np.array(range(n_proc))+1, label=r"Ideal", linewidth=3)
    plt.grid(color = 'grey', linestyle = '-', linewidth = .3)
    plt.xticks(np.array(range(n_proc))+1) # for to show all integers for 1 to 10
    plt.yticks(np.array(range(n_proc))+1) # for to show all integers for 1 to 10
    plt.ylabel("Speedup")
    plt.xlabel(f"Number of processors")
    # plt.xlabel(f"Number of processors, tau = {tau+1}, m = {m}, n = {n}")
    plt.legend()
    plt.savefig(f"images_test/n_{n}_m_{m}_tau_{tau}_acc_{acc}_lbda_{str(lbda)[:6]}_movingseed.png",
                format="png", bbox_inches='tight', dpi=400)

    plt.close()
    with open(f"data_test/data_n_{n}_m_{m}_tau_{tau}_acc_{acc}_lbda_{str(lbda)[:6]}_movingseed.pickle", "wb") as f:
        pickle.dump([list_time_all], f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()