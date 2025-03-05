import numpy as np
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

from functions import block_diag, inv, delete_empty, isConverge
import dynamics as dyn

class Estimator:
    def __init__(self, name, x0_hat=None, P0_hat=None) -> None:
        self.name = name
        self.reset(x0_hat=x0_hat, P0_hat=P0_hat)

    def reset(self, x0_hat, P0_hat):
        self.x_hat = x0_hat
        self.y_hat = None
        self.P_hat = P0_hat

    def estimate(self):
        pass

# calculate (P^-1)*
def cal_Poptim(A:np.ndarray, C:np.ndarray, Q:np.ndarray, R:np.ndarray, B=None, P0=None, gamma=1.0, tol=1e-4) -> np.ndarray:
    '''LQR
    c(x,u) = u.T@R@u + y.T@Q@y
    V(x) = x.T@P@x = c(x,u) + gamma@V'(x)
    '''
    if P0 is None:
        P0 = np.eye(A.shape[0])
    if B is None :
        B = np.eye(A.shape[0])
    P_list = [P0]
    while True:
        P = P_list[-1]
        P = C.T@Q@C + gamma*A.T@P@A - gamma**2*A.T@P@B@inv(R + gamma*B.T@P@B)@B.T@P@A # A.T @ inv(inv(gamma*P) + B@inv(R)@B.T) @ A #
        P_list.append(P)
        # Pinv = A@inv(P + C.T@Q@C)@A.T + inv(R)
        # P_list.append(inv(Pinv))
        if len(P_list) >= 5:
            del P_list[0]
            if isConverge(matrices=P_list, criterion=np.linalg.norm, tol=tol, ord="fro"):
                break
    return P_list[-1]

# Extended Kalman Filter
def EKF(x, P, y_next, model, Q, R) : # 用标称模型做EKF，不感知真实模型
    F = model.F(x=x)
    # predict
    P_pre = F @ P @ F.T
    if Q is not None : P_pre = P_pre + Q
    x_pre, y_pre = model.step(x=x)
    # update
    H = model.H(x=x_pre)
    # P_hat = inv(inv(P_pre) + H.T@inv(R)@H)
    P_hat = P_pre - P_pre@H.T@inv(R+H@P_pre@H.T)@H@P_pre
    x_hat = x_pre - P_hat@H.T@inv(R)@(y_pre - y_next)
    #region 能观性矩阵
    O = np.vstack((H))#, H@F, H@F@F
    if np.linalg.matrix_rank(O) < 1:
        raise ValueError("不可观")
    #endregion
    return x_hat, P_hat

class EKF_class(Estimator):
    def __init__(self, f_fn, h_fn, F_fn, H_fn, x0_hat=None, P0_hat=None) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn
        super().__init__(name="EKF", x0_hat=x0_hat, P0_hat=P0_hat)

    def predict(self, Q, u=None):
        F = self.F_fn(x=self.x_hat, u=u)
        self.x_hat = self.f_fn(x=self.x_hat, u=u)
        self.P_hat = F@self.P_hat@F.T
        if Q.size != 0 : self.P_hat += Q

    def update(self, y, R):
        y_pre = self.h_fn(x=self.x_hat)
        H = self.H_fn(x=self.x_hat)
        K = self.P_hat @ H.T @ inv(H @ self.P_hat @ H.T + R)
        self.x_hat = self.x_hat + K @ (y - y_pre)
        self.P_hat = self.P_hat - self.P_hat@H.T@inv(R+H@self.P_hat@H.T)@H@self.P_hat
        # tempM = inv(inv(self.P_hat) + H.T@inv(R)@H)
        # gamma = max(np.linalg.eigvals(tempM))*1.2
        # self.P_hat = inv(inv(self.P_hat) + H.T@inv(R)@H - 1/gamma*np.eye(2))
        self.y_hat = self.h_fn(x=self.x_hat)

    def estimate(self, y, Q, R, u=None):
        self.predict(Q=Q, u=u)
        self.update(y=y, R=R)
        # if self.x_hat.size == 4: 
        #     self.x_hat[0] = 0
        #     self.x_hat[1] = 0

class GKF_class(Estimator): # 用于广义系统的KF
    def __init__(self, f_fn, h_fn, F_fn, H_fn, x0_hat=None, P0_hat=None) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn
        super().__init__(name="EKF", x0_hat=x0_hat, P0_hat=P0_hat)

    def estimate(self, y, Q, R, u=None):
        if not hasattr(self, "E") :
            n = Q.shape[0]
            non_zeros = [i for i in range(n) if np.linalg.norm(Q[i]) != 0]
            r = len(non_zeros)
            E = np.zeros((r,n))
            for i in range(r) : E[i, non_zeros[i]] = 1
            self.E = E
        E = self.E
        A = E @ self.F_fn()
        C = self.H_fn()
        Q = E @ Q @ E.T
        self.P_hat = E.T @ inv(Q + A @ self.P_hat @ A.T) @ E + C.T @ R @ C
        self.P_hat = inv(self.P_hat)
        self.x_hat = E.T @ A @ self.x_hat - self.P_hat @ C.T @ inv(R) @ (C @ E.T @ A @ self.x_hat - y)
        self.y_hat = self.h_fn(x=self.x_hat)

class MHE(Estimator):
    def __init__(self, f_fn, h_fn, F_fn, H_fn, window, x0_hat=None, P0_hat=None) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn
        self.window = window
        super().__init__(name="MHE", x0_hat=x0_hat, P0_hat=P0_hat)

    def reset(self, x0_hat=None, P0_hat=None):
        self.x_hat = x0_hat
        if x0_hat is not None : self.dim_state = x0_hat.size
        self.P_hat = P0_hat
        self.y_seq = []
        self.u_seq = []
        self.x0_bar_seq = [x0_hat]

    def estimate(self, y, Q, R, u=None, P_inv=None):
        self.y_seq.append(y)
        self.u_seq.append(u)
        if len(self.y_seq) > self.window : 
            del self.y_seq[0]
            del self.u_seq[0]
        if P_inv is None : P_inv = inv(self.P_hat)
        result = NLSF_uniform(P_inv=P_inv, y_seq=self.y_seq, u_seq=self.u_seq, Q=Q, R=R, f=self.f_fn, h=self.h_fn, F=self.F_fn, H=self.H_fn, 
                              mode="quadratic", x0=self.x0_bar_seq[:], x0_bar=self.x0_bar_seq[0])
        self.x_hat = result.x[-self.dim_state:]
        self.y_hat = self.h_fn(x=self.x_hat)
        # EKF方法更新P(直观解释：x0被删去的时候才需要更新P)
        if len(self.x0_bar_seq) == self.window: 
            x0_hat = self.x0_bar_seq[0]
            x1_pre = self.f_fn(x=x0_hat) # 先不加u了很麻烦，有u的话再说吧
            F = self.F_fn(x=x0_hat)
            P_pre = F@self.P_hat@F.T + Q
            H = self.H_fn(x=x1_pre)
            self.P_hat = P_pre - P_pre@H.T@inv(R+H@P_pre@H.T)@H@P_pre
        # 更新x0_bar_seq
        # self.x0_bar_seq = list(result.x.reshape(-1, self.dim_state)) # 要用新的就都用新的
        self.x0_bar_seq.append(self.x_hat) # 要不用新的就都不用新的，训练的时候是不用新的所以测试也不应该用新的
        if len(self.x0_bar_seq) > self.window : del self.x0_bar_seq[0]

# def IEKF(x, P, y_next, Q, R, times=10):
#     # predict
#     F = dyn.F(x)
#     P_pre = F @ P @ F.T
#     if Q.size != 0 : P_pre = P_pre + Q
#     x_pre0 = dyn.f(x)
#     x_pre = x_pre0
#     # update
#     for _ in range(times):
#         H = dyn.H(x=x_pre)
#         y_pre = dyn.h(x=x_pre)
#         P_hat = inv(inv(P_pre) + H.T@inv(R)@H)
#         x_hat = x_pre0 - (P_hat@H.T@inv(R)@(y_pre - y_next).T).T
#         x_hat = np.squeeze(x_hat)
#         x_pre = x_hat
#     return x_hat, P_hat

# Unscented Kalman Filter
# def UKF(state, P, obs_next, Q, R, alpha=.5, beta=2., kappa=-5.) : 
#     n = state.size
#     nw = Q.shape[1]
#     nv = R.shape[1]
#     na = n + nw + nv
#     lamda = alpha**2 * (na + kappa) - na

#     # calculate sigma points and weights
#     xa = np.hstack((state, np.zeros((nw, )), np.zeros((nv, ))))
#     xa_sigma = np.tile(xa, (2*na+1, 1))
#     M = (na+lamda)*block_diag([P, Q, R])
#     M = np.linalg.cholesky(M)
#     xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
#     xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
#     xx_sigma = xa_sigma[:, :n]
#     xw_sigma = xa_sigma[:,n:n+nw]
#     xv_sigma = xa_sigma[:,n+nw: ]
#     Wc = np.ones((2*na+1, )) * 0.5 / (na+lamda)
#     Wm = np.ones((2*na+1, )) * 0.5 / (na+lamda)
#     Wc[0] = lamda / (na + lamda) + 1 - alpha**2 + beta
#     Wm[0] = lamda / (na + lamda)

#     # time update
#     x_next_pre = dyn.f(xx_sigma)
#     x_next_pre_aver = np.average(x_next_pre, weights=Wm, axis=0)
#     P_next_pre = np.zeros((n,n))
#     for i in range(2*na+1) : 
#         P_next_pre += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (x_next_pre[i] - x_next_pre_aver).reshape(1,-1)
#     P_next_pre += Q
    
#     # resample sigma points
#     xa = np.hstack((x_next_pre_aver, np.zeros((nw, )), np.zeros((nv, ))))
#     xa_sigma = np.tile(xa, (2*na+1, 1))
#     M = (na+lamda)*block_diag([P_next_pre, Q, R])
#     M = np.linalg.cholesky(M)
#     xa_sigma[1:na+1] = xa_sigma[1:na+1] + M
#     xa_sigma[na+1: ] = xa_sigma[na+1: ] - M
#     xx_sigma = xa_sigma[:, :n]
#     xw_sigma = xa_sigma[:,n:n+nw]
#     xv_sigma = xa_sigma[:,n+nw: ]

#     # measurement update ## 有一种是直接用上面的sigma点做y的预测的，还有一种是用上面算出来的x_pre_aver和P_pre重新选择sigma点做y预测的，下面先采用前者简单方式
#     y_next_pre = dyn.h(xx_sigma)
#     y_next_pre_aver = np.average(y_next_pre, weights=Wm, axis=0)
#     P_yy = np.zeros_like(R)
#     P_xy = np.zeros((n, nv))
#     for i in range(2*na+1) : 
#         P_yy += Wc[i] * (y_next_pre[i] - y_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
#         P_xy += Wc[i] * (x_next_pre[i] - x_next_pre_aver).reshape(-1,1) @ (y_next_pre[i] - y_next_pre_aver).reshape(1,-1)
#     P_yy += R
#     K = P_xy @ inv(P_yy)
#     x_next_hat = x_next_pre_aver + K @ (obs_next - y_next_pre_aver)
#     P_next_hat = P_next_pre - K @ P_yy @ K.T

#     return x_next_hat.reshape(-1), P_next_hat


def NLSF_uniform(P_inv, y_seq, u_seq, Q, R, f, h, F, H, mode:str="quadratic", x0=None, **args) : 
    if "sumofsquares" in mode.lower() : 
        fun = SumOfSquares(f_fn=f, h_fn=h, F_fn=F, H_fn=H)
        params = [P_inv, y_seq, u_seq, Q, R]
    elif "quadratic" in mode.lower() : 
        fun = Quadratic(f_fn=f, h_fn=h, F_fn=F, H_fn=H)
        params = [P_inv, y_seq, u_seq, Q, R, args["x0_bar"]]

    if "gamma" in args.keys() : params.append(args["gamma"])
    if "end" in mode.lower() : params.append(args["xend"])

    ds = Q.shape[0]
    if x0 is None : 
        x0 = np.zeros((ds*(len(y_seq)+1), ))
    else : 
        while (len(x0) <= len(y_seq)) : 
            x0.append(f(x0[-1]))
        x0 = np.array(x0).reshape(-1)
    result = least_squares(fun.res_fun, x0, method='lm', jac=fun.jac_fun, args=params) # , max_nfev=8
    return result

class SumOfSquares() : 
    def __init__(self) -> None:
        pass

    def res_fun(self, x, LP, y_seq, Q, R, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))

        LQ = np.linalg.cholesky(inv(Q))
        LR = np.linalg.cholesky(inv(R))
        f = np.insert(x[:ds], 0, 1)[np.newaxis,:]
        L = LP[:]
        for i in range(num_obs) : 
            f = np.hstack((f, x[ds*(i+1):ds*(i+2)]-dyn.f(x[ds*(i):ds*(i+1)])[np.newaxis,:], 
                              y_seq[i]-dyn.h(x[ds*(i+1):ds*(i+2)])[np.newaxis,:]))
            L = block_diag((L, LQ, LR))
        
        if xend is not None : 
            f = np.hstack((f, (xend-dyn.f(x[-ds:]))[np.newaxis,:]))
            L = block_diag((L, LQ))
        
        return (f@L).reshape(-1)

    def jac_fun(self, x, LP, y_seq, Q, R, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : np.vstack((np.hstack((-dyn.F(x0), np.eye(ds))), np.pad(-dyn.H(x1), ((0,0),(ds,0)))))

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
            Jadd = np.pad(-dyn.F(xend), ((0,0),(ds*num_obs,0)))
            J = np.vstack((J, Jadd))
            L = block_diag((L, LQ))

        return L.T@J

class Quadratic() : 
    def __init__(self, f_fn, h_fn, F_fn, H_fn) -> None:
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.F_fn = F_fn
        self.H_fn = H_fn

    def res_fun(self, x, P_inv, y_seq, u_seq, Q, R, x0_bar, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = x.size // (num_obs+1)

        f = np.array(x[:ds] - x0_bar)[np.newaxis,:]
        M = np.copy(P_inv)
        for i in range(num_obs) : 
            if u_seq[i] is None : u_seq[i] = 0
            f = np.hstack((f, (x[ds*(i+1):ds*(i+2)]-self.f_fn(x[ds*(i):ds*(i+1)])-u_seq[i])[np.newaxis,:], 
                              (y_seq[i]-self.h_fn(x[ds*(i+1):ds*(i+2)]))[np.newaxis,:]))
            M = block_diag((M * gamma, inv(Q), inv(R)))
        
        if xend is not None : 
            f = np.hstack((f, (xend-self.f_fn(x[-ds:]))[np.newaxis,:]))
            M = block_diag((M * gamma, inv(Q)))

        # 如果M有全零行，把该行以及行号相同的列剔除并且剔除f中对应的元素
        non_zeros = ~np.all(M == 0, axis=1)
        M = M[non_zeros]
        M = M[:, non_zeros]
        f = f[:, non_zeros]
        
        L = np.linalg.cholesky(M)
        return (f@L).reshape(-1)

    def jac_fun(self, x, P_inv, y_seq, u_seq, Q, R, x0_bar, gamma=1.0, xend=None) : 
        num_obs = len(y_seq)
        ds = int(x.size / (num_obs+1))
        jadd = lambda x0, x1 : np.vstack((np.hstack((-self.F_fn(x0), np.eye(ds))), np.pad(-self.H_fn(x1), ((0,0),(ds,0)))))

        J = np.eye(ds)
        M = P_inv[:]
        for i in range(num_obs) : 
            J = np.pad(J, ((0,0), (0,ds)))
            Jadd = np.pad(jadd(x[ds*i:ds*(i+1)], x[ds*(i+1):ds*(i+2)]), ((0,0), (i*ds,0)))
            J = np.vstack((J, Jadd))
            M = block_diag((M * gamma, inv(Q), inv(R)))

        if xend is not None : 
            Jadd = np.pad(-self.F_fn(xend), ((0,0),(ds*num_obs,0)))
            J = np.vstack((J, Jadd))
            M = block_diag((M * gamma, inv(Q)))

        # 如果M有全零行，把该行以及行号相同的列剔除并且剔除J中对应的行
        non_zeros = ~np.all(M == 0, axis=1)
        M = M[non_zeros]
        M = M[:, non_zeros]
        J = J[non_zeros]

        L = np.linalg.cholesky(M).T
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