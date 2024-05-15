import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

from functions import block_diag, inv, delete_empty
from params import GP
import dynamics as dyn

# Extended Kalman Filter
# class Estimator :
#     def __init__(self, f_fn, h_fn, Q, R, x0_hat, P0_hat) -> None:
#         self.f_fn = f_fn
#         self.h_fn = h_fn
#         self.Q = Q
#         self.R = R
#         self.xt_hat = torch.tensor(x0_hat, dtype=torch.float32, device=GP.device)
#         self.Pt_hat = torch.tensor(P0_hat, dtype=torch.float32, device=GP.device)
#     def predict(self):
#         pass
#     def update(self, yt_p1):
#         pass
#     def estimate(self, yt_p1):
#         self.predict()
#         self.update(yt_p1=yt_p1)

# class EKF(Estimator):
#     def __init__(self, f_fn, h_fn, Q, R, x0_hat, P0_hat) -> None:
#         super().__init__(f_fn, h_fn, Q, R, x0_hat, P0_hat)
    
#     def predict(self):
#         self.xt_p1_pre = self.f_fn(self.xt_hat)
#         self.xt_p1_pre.backward()
#         F = self.xt_hat.grad()
#         self.Pt_p1_pre = F @ self.Pt_hat @ F.T
#         if self.Q.size != 0 : self.Pt_p1_pre = self.Pt_p1_pre + self.Q
    
#     def update(self, yt_p1):
#         yt_p1_pre = 
#         pass
def EKF (x, P, y_next, Q, R) : 
    # linearization system matrix 1 #####################
    F = dyn.F_fn(x)
    # ###################################################

    # predict
    P_pre = F @ P @ F.T
    if Q.size != 0 : P_pre = P_pre + Q
    x_pre = dyn.f_fn(x)
    # update
    y_pre = dyn.h_fn(x_pre)
    H = dyn.H_fn(x_pre)
    P_hat = inv(inv(P_pre) + H.T@inv(R)@H)
    x_hat = x_pre - (P_hat@H.T@inv(R)@(y_pre - y_next).T).T

    return x_hat.reshape((-1,)), P_hat

# Unscented Kalman Filter
def UKF(state, P, obs_next, Q, R, alpha=.5, beta=2., kappa=-5.) : 
    n = state.size
    nw = Q.shape[1]
    nv = R.shape[1]
    na = n + nw + nv
    lamda = alpha**2 * (na + kappa) - na

    # calculate sigma points and weights
    xa = np.hstack((state, np.zeros((nw, )), np.zeros((nv, ))))
    xa_sigma = np.tile(xa, (2*na+1, 1))
    M = (na+lamda)*block_diag([P, Q, R])
    M = np.linalg.cholesky(M)
    xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
    xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
    xx_sigma = xa_sigma[:, :n]
    xw_sigma = xa_sigma[:,n:n+nw]
    xv_sigma = xa_sigma[:,n+nw: ]
    Wc = np.ones((2*na+1, )) * 0.5 / (na+lamda)
    Wm = np.ones((2*na+1, )) * 0.5 / (na+lamda)
    Wc[0] = lamda / (na + lamda) + 1 - alpha**2 + beta
    Wm[0] = lamda / (na + lamda)

    # time update
    x_next_pre = dyn.f_fn(xx_sigma)
    x_next_pre_aver = np.average(x_next_pre, weights=Wm, axis=0)
    P_next_pre = np.zeros((n,n))
    for i in range(2*na+1) : 
        P_next_pre += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (x_next_pre[i] - x_next_pre_aver).reshape(1,-1)
    P_next_pre += Q
    
    # resample sigma points
    xa = np.hstack((x_next_pre_aver, np.zeros((nw, )), np.zeros((nv, ))))
    xa_sigma = np.tile(xa, (2*na+1, 1))
    M = (na+lamda)*block_diag([P_next_pre, Q, R])
    M = np.linalg.cholesky(M)
    xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
    xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
    xx_sigma = xa_sigma[:, :n]
    xw_sigma = xa_sigma[:,n:n+nw]
    xv_sigma = xa_sigma[:,n+nw: ]

    # measurement update ## 有一种是直接用上面的sigma点做y的预测的，还有一种是用上面算出来的x_pre_aver和P_pre重新选择sigma点做y预测的，下面先采用前者简单方式
    y_next_pre = dyn.h_fn(xx_sigma)
    y_next_pre_aver = np.average(y_next_pre, weights=Wm, axis=0)
    P_yy = np.zeros_like(R)
    P_xy = np.zeros((n, nv))
    for i in range(2*na+1) : 
        P_yy += Wc[i] * (y_next_pre[i] - y_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
        P_xy += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
    P_yy += R
    K = P_xy @ inv(P_yy)
    x_next_hat = x_next_pre_aver + K @ (obs_next - y_next_pre_aver)
    P_next_hat = P_next_pre - K @ P_yy @ K.T

    return x_next_hat.reshape(-1), P_next_hat


def NLSF_uniform(P_inv, y_seq, Q, R, mode:str="quadratic", x0=None, **args) : 
    if "sumofsquares" in mode.lower() : 
        fun = SumOfSquares()
        params = [P_inv, y_seq, Q, R]
    elif "quadratic" in mode.lower() : 
        fun = Quadratic()
        params = [P_inv, y_seq, Q, R, args["x0_bar"]]
    
    if "end" in mode.lower() : params.append(args["xend"])

    ds = Q.shape[0]
    if x0 is None : 
        x0 = torch.zeros((ds*(len(y_seq)+1), ))
    else : 
        while (len(x0) <= len(y_seq)) : 
            x0.append(dyn.f_fn(x0[-1]))
        x0 = torch.stack(x0).reshape((-ds,))
    result = least_squares(fun.res_fun, x0, method='lm', jac=fun.jac_fun, args=params) # , max_nfev=8
    return result

class SumOfSquares() : 
    def __init__(self) -> None:
        pass

    def res_fun(self, x, LP, y_seq, Q, R, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.shape[0] / (num_obs+1))

        LQ = torch.linalg.cholesky(inv(Q))
        LR = torch.linalg.cholesky(inv(R))
        f = np.insert(x[:ds], 0, 1)[np.newaxis,:]
        L = LP[:]
        for i in range(num_obs) : 
            f = torch.hstack((f, x[ds*(i+1):ds*(i+2)]-dyn.f_fn(x[ds*(i):ds*(i+1)])[np.newaxis,:], 
                              y_seq[i]-dyn.h_fn(x[ds*(i+1):ds*(i+2)])[np.newaxis,:]))
            L = block_diag((L, LQ, LR))
        
        if xend is not None : 
            f = np.hstack((f, (xend-dyn.f_fn(x[-ds:]))[np.newaxis,:]))
            L = block_diag((L, LQ))
        
        return (f@L).reshape(-1)

    def jac_fun(self, x, LP, y_seq, Q, R, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : np.vstack((np.hstack((-dyn.F_fn(x0), np.eye(ds))), np.pad(-dyn.H_fn(x1), ((0,0),(ds,0)))))

        LQ = np.linalg.cholesky(inv(Q))
        LR = np.linalg.cholesky(inv(R))
        J = np.pad(np.eye(ds), ((1,0),(0,0)))
        L = LP[:]
        for i in range(num_obs) : 
            J = np.pad(J, ((0,0), (0,ds)))
            Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
            J = np.vstack((J, Jadd))
            L = block_diag((L, LQ, LR))

        if xend is not None : 
            Jadd = np.pad(-dyn.F_fn(xend), ((0,0),(ds*num_obs,0)))
            J = np.vstack((J, Jadd))
            L = block_diag((L, LQ))

        return L.T@J

class Quadratic() : 
    def __init__(self) -> None:
        pass

    def res_fun(self, x, P_inv, y_seq, Q, R, x0_bar, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.shape[0] / (num_obs+1))

        f = np.tile(np.array(x[:ds] - x0_bar), (1,1))
        M = P_inv[:]
        for i in range(num_obs) : 
            f = np.hstack((f, x[ds*(i+1):ds*(i+2)]-dyn.f_fn(x[ds*(i):ds*(i+1)])[np.newaxis,:], 
                              y_seq[i]-dyn.h_fn(x[ds*(i+1):ds*(i+2)])[np.newaxis,:]))
            M = block_diag((M, inv(Q), inv(R)))
        
        if xend is not None : 
            f = np.hstack((f, (xend-dyn.f_fn(x[-ds:]))[np.newaxis,:]))
            M = block_diag((M, inv(Q)))
        
        L = np.linalg.cholesky(M)
        return (f@L).reshape(-1)

    def jac_fun(self, x, P_inv, y_seq, Q, R, x0_bar, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : torch.vstack((torch.hstack((-dyn.F_fn(x0), torch.eye(ds))), torch.nn.functional.pad(-dyn.H_fn(x1), ((0,0),(ds,0)))))

        J = torch.eye(ds)
        M = P_inv[:]
        for i in range(num_obs) : 
            J = torch.nn.functional.pad(J, ((0,0), (0,ds)))
            Jadd = torch.nn.functional.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
            J = torch.vstack((J, Jadd))
            M = block_diag((M, inv(Q), inv(R)))

        if xend is not None : 
            Jadd = torch.nn.functional.pad(-dyn.F_fn(xend), ((0,0),(ds*num_obs,0)))
            J = torch.vstack((J, Jadd))
            M = block_diag((M, inv(Q)))

        L = torch.linalg.cholesky(M).T
        return L@J


class Particle_Filter() : 
    def __init__(self, state_dim:int, obs_dim:int, num_particles:int, fx, hx, x0_mu, P0, threshold=None, rand_num=1111) -> None:
        self.state_dim = state_dim
        self.obs_dim   = obs_dim
        self.N         = num_particles
        self.fx        = fx
        self.hx        = hx
        self.threshold = self.N * 0.5 if threshold is None else threshold
        np.random.seed(seed=rand_num)
        self.create_gaussian_particles(x0_mu, P0, self.N)

    def create_uniform_particles(self, state_dim, state_range, N) : 
        self.particles = np.empty((N, state_dim))
        for i in range(state_dim) : 
            self.particles[:, i] = np.random.uniform(state_range[i][0], state_range[i][1], size=N)
        self.weight = np.ones((N, ))/N

    def create_gaussian_particles(self, mean, cov, N) : 
        self.particles = np.random.multivariate_normal(mean, cov, N)
        self.weight = np.ones((N, ))/N
    
    def predict(self, noise_Q, noise_mu=None, dt=.1) : 
        if noise_mu is None : noise_mu = np.zeros((self.state_dim, ))
        process_noise = np.random.multivariate_normal(noise_mu, noise_Q, self.N)
        self.particles = self.fx(self.particles, process_noise, dt)

    def update(self, observation, obs_noise_R) : 
        for i in range(self.N) : 
            self.weight[i] *= multivariate_normal(self.hx(self.particles[i]), obs_noise_R).pdf(observation)
        
        self.weight += 1.e-300          # avoid round-off to zero
        self.weight /= sum(self.weight) # normalize

    def estimate(self) : 
        state_hat = np.average(self.particles, weights=self.weight, axis=0)
        Cov_hat = np.cov(self.particles, rowvar=False, aweights=self.weight)

        if self.neff() >= self.threshold : 
            print(f'resample, Wneff={self.neff()}')
            self.simple_resample()
        return state_hat, Cov_hat
    
    def simple_resample(self) : 
        cumulative_sum = np.cumsum(self.weight)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))

        # resample according to indexes
        self.particles = self.particles[indexes]
        self.weight = np.ones((self.N, ))/self.N

    def neff(self) : 
        return 1. / np.sum(np.square(self.weight))