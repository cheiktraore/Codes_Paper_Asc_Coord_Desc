# Run before importing numpy to force each process running single-threaded
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

A, y, x_true = make_correlated_data(n, m, density=s, random_state=0)

lbda_max = norm(A.T @ y, ord=np.inf)
lbda = (lbda_max / 100)

l_res = np.max(norm(A.T@A, axis=0))

clf = Lasso(alpha=lbda/len(y), fit_intercept=False,
            max_iter=20000, tol=1e-10)
clf.fit(A, y)
f_star = lasso_loss(A, y, lbda, clf.coef_)
duration = 120
max_iter = int(6e7)

def asyn_prox_cd(A, y, lbda, x, fx, gammas, f_star, start_time, counter):
    while time.perf_counter() - start_time < duration:
        i = np.random.randint(m)
        neg_grad_i = A[:, i] @ (y - A @ x )
        new = ST(x[i] + neg_grad_i * gammas[i], lbda * gammas[i])
        x[i] = new
        fx[counter[0]] = lasso_loss(A, y, lbda, x) - f_star
        counter[0] += 1

def syn_prox_cd(A, y, lbda, x, gammas, proc):
    neg_grad_i = A[:, proc] @ (y - A @ x )
    new = ST(x[proc] + neg_grad_i * gammas[proc], lbda * gammas[proc])
    x[proc] = new
        
if __name__ == '__main__':
###############  Algo 1.1 #################################################################
    np.random.seed(0)
    N_PROC = 10
    x = np.zeros(m)
    step = .5*( norm(A, axis=0)**2 + (2*(N_PROC)*l_res)/np.sqrt(m))
    gammas = 1. / (step)
    fx = np.zeros(max_iter)
    counter = np.zeros(1, dtype=np.int64)
    with SharedMemoryManager() as smm:
        shm_x = smm.SharedMemory(size=x.nbytes)
        # Now create a NumPy array backed by shared memory
        x_shared = np.ndarray(x.shape, dtype=x.dtype, buffer=shm_x.buf)
        x_shared[:] = x[:]
        
        shm_count = smm.SharedMemory(size=counter.nbytes)
        # Now create a NumPy array backed by shared memory
        counter_shared = np.ndarray(counter.shape, dtype=counter.dtype, buffer=shm_count.buf)
        counter_shared[:] = counter[:]

        shm_fx = smm.SharedMemory(size=fx.nbytes)
        fx_shared = np.ndarray(fx.shape, dtype=fx.dtype, buffer=shm_fx.buf)
        fx_shared[:] = fx[:]
        
        start = time.perf_counter()
        processes = [mp.Process(target=asyn_prox_cd, args=(A, y, lbda, x_shared, fx_shared, 
                                                           gammas, f_star, start, 
                                                           counter_shared)) 
                     for _ in range(N_PROC)]

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        for proc in processes:
            proc.terminate()
            
    shm_x.close()   # Close each SharedMemory instance
    shm_count.close()
    fx_async = fx_shared[fx_shared > 0]
    shm_fx.close()

############################ Synchronous ######################################
    x_sync = np.zeros(m)

    tau = 10
    step = .5*tau*( norm(A, axis=0)**2 )

    gammas = 1. / (step)
    fx_sync = []
    # it  = 0
    count = 0
    init_time = 0

    with SharedMemoryManager() as smm:
        shm_x_sync = smm.SharedMemory(size=x_sync.nbytes)
        # Now create a NumPy array backed by shared memory
        x_shared_sync = np.ndarray(x_sync.shape, dtype=x_sync.dtype, buffer=shm_x_sync.buf)
        x_shared_sync[:] = x_sync[:]

        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            indices = np.random.choice(np.array(range(m)), N_PROC, replace=False)
            processes = []
            for proc in range(N_PROC):
                processes.append(mp.Process(target=syn_prox_cd, args=(A, y, lbda, 
                                                                      x_shared_sync, gammas, 
                                                                      indices[proc])))

            for proc in processes:
                proc.start()
                
            for proc in processes:
                proc.join()

            fx_sync.append(lasso_loss(A, y, lbda, x_shared_sync) - f_star)

            for proc in processes:
                proc.terminate()
                
        shm_x_sync.close()

########################## Plot #############################################
    plt.figure(constrained_layout=True)
    normalizer = len(fx_sync) / fx_async.shape[0]
    plt.semilogy(normalizer*np.arange(fx_async.shape[0]), fx_async,
             label=r"Algo 1.1", linewidth=3)
    plt.semilogy(range(len(fx_sync)), fx_sync, label=r"Synchronous", linewidth=3)
    plt.ylabel("$F(\\mathbf{x}^k) - F^*$")
    plt.xlabel(f"Iterations")
    plt.legend()
    plt.savefig(f"images_test/sync_n_{n}_m_{m}_time_{duration}_step_{tau}.png",
                format="png", bbox_inches='tight', dpi=400)
    plt.close()
    
    with open(f"data_test/sync_n_{n}_m_{m}_time_{duration}_step_{tau}.pickle", "wb") as f:
        pickle.dump([fx_async, fx_sync], f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()