import numpy as np
import matplotlib.pyplot as plt
import pickle

import dynamics as dyn
from estimator import EKF
from simulate import simulate
from model import create_model
from plot import plotReward
from params import def_param2, set_params
from estimator import NLSF_uniform, cal_Poptim
from functions import inv, RandomGenerator, vectorize, vec2mat, isConverge, ds2do

#region 设置绘图中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 28
#endregion

class LSTDO:
    '''Linear Observer Learning by Temporal Difference'''
    def __init__(self, dim_state, dim_obs, x0_hat, f_fn, h_fn, gamma=0.9, randSeed=11111) -> None:
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.x_hat = x0_hat
        self.x0_hat = x0_hat
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.gamma = gamma
        np.random.seed(randSeed)
        self.H = np.diag(np.abs(np.random.multivariate_normal(mean=np.zeros(dim_state), cov=100*np.eye(self.dim_state))))
        self.h = np.random.normal(loc=0, scale=10)
        self.randomGen = RandomGenerator(randomFun=np.random.multivariate_normal, rand_num=randSeed)

    def train(self, y_batch, Q, R, epsilon=1e3):
        #region Ptest
        Poptim_inv = np.array([[ 0.63128271,  0.35470852],
       [ 0.35470852, 10.56484081]])
        H_abserror_batch = [np.linalg.norm(self.H - Poptim_inv, ord='fro')]
        # h_abserror_batch = []
        delta_batch = []
        # if converge
        params_batch = []
        #endregion
        for i in range(len(y_batch)):
            y_seq = y_batch[i]
            self.reset(x0_hat=self.x0_hat, P0_hat=None)
            params = np.insert(vectorize(self.H), obj=-1, values=self.h).reshape((-1,1)) # 递归LS
            A = 0 # LS
            b = 0 # LS
            t_seq = range(len(y_seq))
            x_hat_seq = [self.x_hat]
            delta_batch.append(0)
            ksi = self.randomGen.getRandomList(length=len(y_seq), mean=np.zeros(self.dim_state), cov=epsilon*np.eye(self.dim_state))
            P = np.eye(ds2do(self.dim_state))
            for t in t_seq:
                result = NLSF_uniform(P_inv=self.H, y_seq=[y_seq[t]], Q=Q, R=R, mode="quadratic", x0=[self.x_hat], gamma=self.gamma, x0_bar=self.x_hat)
                self.x_hat = result.x[-self.dim_state:]
                x_hat_seq.append(self.x_hat)
                x_next_noise = self.x_hat+ksi[t]
                result = NLSF_uniform(P_inv=self.H, y_seq=[], Q=Q, R=R, mode="quadratic-end", x0=x_hat_seq[t], gamma=self.gamma, x0_bar=x_hat_seq[t], xend=x_next_noise)
                Voptim = result.fun@result.fun + (y_seq[t] - self.h_fn(x_next_noise))@inv(R)@(y_seq[t] - self.h_fn(x_next_noise)) + self.h
                delta_batch[-1] += np.abs(Voptim - ksi[t].reshape((1,-1))@ksi[t].reshape((-1,1)) - self.h).item()
                gradient = vectorize(ksi[t].reshape((-1,1))@ksi[t].reshape((1,-1)))
                gradient = np.concatenate((gradient, np.ones((1,)))).reshape((1,-1))
                #region 递归最小二乘
                P = P - P@gradient.T@gradient@P/(1+(gradient@P@gradient.T).item())
                params = params + P@gradient.T@(Voptim - gradient@params)
                params_batch.append(params[:-1])
                if len(params_batch) > 5 :
                    del params_batch[0]
                    if isConverge(params_batch, tol=1e-6) :
                        self.H = vec2mat(params.flatten()[:-1])
                        self.h = params[-1].item()
                        break
            self.H = vec2mat(params.flatten()[:-1])
            self.h = params[-1].item()
            #endregion
            #region 最小二乘
            #     b += Voptim*gradient.T
            #     A += gradient.T@gradient
            # newParams = np.linalg.solve(a=A, b=b)
            # self.H = vec2mat(newParams.flatten()[:-1])
            # self.h = newParams[-1].item()
            #endregion
            H_abserror_batch.append(np.linalg.norm(self.H - Poptim_inv, ord='fro'))
            # h_abserror_batch.append(np.abs(self.h, ))
        #region plot error
        # H误差
        fig, ax = plt.subplots()
        plt.yscale('log')
        plt.grid(True)
        ax.plot(range(len(H_abserror_batch)), H_abserror_batch, 'o', color='r', linestyle='--', label="value")
        ax.plot(range(len(H_abserror_batch)), np.average(H_abserror_batch)*np.ones_like(H_abserror_batch), color='r', label="average")
        ax.set_xlim(0, len(H_abserror_batch))
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('||H-P$^{*-1}$||$_F$')
        ax.set_title('(H-P$^{*-1}$)的Frobenius范数')
        ax.legend()
        # δ误差
        fig, ax = plt.subplots()
        plt.yscale('log')
        plt.grid(True)
        ax.plot(range(len(delta_batch)), delta_batch, 'o', color='r', linestyle='--', label="value")
        ax.plot(range(len(delta_batch)), np.average(delta_batch)*np.ones_like(delta_batch), color='r', label="average")
        ax.set_xlim(0, len(delta_batch))
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('|δ|')
        ax.set_title('Absolute value of δ')
        ax.legend()
        plt.show()
        #endregion

    def estimate(self, y, Q, R): #gamma 会不会起到作用？
        result = NLSF_uniform(P_inv=self.H, y_seq=[y], Q=Q, R=R, mode="quadratic", x0=[self.x_hat], gamma=self.gamma, x0_bar=self.x_hat)
        self.x_hat = result.x[-self.dim_state:]

        # try:
        #     x_pre = self.f_fn(x=self.x_hat)
        #     y_pre = self.h_fn(x=x_pre)
        #     K = self.K
        # except:
        #     F = dyn.F(self.x_hat)
        #     # predict
        #     P_pre = F @ inv(self.H) @ F.T
        #     if Q.size != 0 : P_pre = P_pre + Q
        #     x_pre, y_pre = dyn.step(self.x_hat)
        #     # update
        #     H = dyn.H(x_pre)
        #     # P_hat = inv(P_pre_inv + H.T@inv(R)@H)
        #     P_hat = P_pre - P_pre@H.T@inv(R+H@P_pre@H.T)@H@P_pre
        #     K = P_hat@H.T@inv(R)
        #     self.K = K
        # x_hat = x_pre - (K@(y_pre - y).T).T
        # self.x_hat = np.squeeze(x_hat)

    def reset(self, x0_hat, P0_hat):
        self.x_hat = x0_hat

class RL_Observer:
    '''Optimal Observer Design Using Reinforcement Learning and Quadratic Neural Networks'''
    def __init__(self, dim_state, dim_obs, x0_hat, P0_hat, f_fn, h_fn, gamma=0.6) -> None:
        # 不变
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.x0_hat = x0_hat
        self.P0_hat = P0_hat
        self.f_fn = f_fn
        self.h_fn = h_fn
        self.gamma = gamma
        np.random.seed(111)
        # 可变
        self.x_hat = x0_hat
        self.P_hat = P0_hat
        self.xhat_list = [self.x_hat] # 从xhat0开始
        self.y_list = [] # 从y_1开始
        self.H = 5*np.eye(N=self.dim_state*(self.dim_state+self.dim_obs))

    def train(self, y_batch, Q, R) -> None:
        ds = self.dim_state
        do = self.dim_obs
        #region Ptest
        H_optim = self.cal_Hoptim(Q=1*Q, R=1*R)
        # self.H = self.cal_Hoptim(Q=5*Q, R=5*R)
        H_abserror_batch = [np.linalg.norm(self.H[:ds] - H_optim[:ds], ord='fro')]
        MT_leftinv = inv(self.M@self.M.T)@self.M
        M_rightinv = self.M.T@inv(self.M@self.M.T)
        # h_abserror_batch = []
        delta_batch = []
        # is converge
        params_batch = []
        params_seq = []
        #endregion
        # for i in range(len(y_batch)):
        i = 0
        while True:
            i += 1
            # y_seq = y_batch[i]
            self.reset(x0_hat=self.x0_hat, P0_hat=self.P0_hat)
            # t_seq = range(len(y_seq))
            x_hat_seq = [self.x_hat]
            # params = vectorize(self.H).reshape((-1,1)) # 递归LS
            params = self.H[:ds].reshape((-1,1)) # 递归LS(2)
            # A_LS = 0 # 批LS
            # b_LS = 0 # 批LS
            P = 10*np.eye(params.shape[0])
            update_flag = True
            # for t in t_seq:
            #     y = y_seq[t]
            x = np.random.multivariate_normal(mean=np.array([0, 0]), cov=np.diag((1., 1.)))
            t = 0
            while True:
                x = self.f_fn(x=x, u=None, disturb=None)
                y = self.h_fn(x=x, noise=None)
                t += 1
                if len(self.y_list) < ds + 1:
                    # 小于nx维的时候，用EKF
                    self.x_hat, self.P_hat = EKF(x=self.x_hat, P=self.P_hat, y_next=y, Q=Q, R=R)
                    self.xhat_list.append(self.x_hat)
                    self.y_list.append(y)
                else : 
                    H_11 = self.H[0:ds, 0:ds]
                    H_omega = self.H[0:ds, ds:ds*ds]
                    H_y = self.H[0:ds, ds*ds:]
                    
                    # self.xhat_list 从k-nx 到 k+1
                    # self.y_list    从k-nx+1 到 k+1
                    xpre_list = [self.f_fn(x=x_hat) for x_hat in self.xhat_list[:-1]] # 从t=k-nx+1到k+1，xpre_1 = self.f_fn(xhat_0)
                    omega_list = [(x_hat - x_pre) for x_hat, x_pre in zip(self.xhat_list[1:], xpre_list)] # 从t=k-nx到k，omega_0 = xhat_1 - xpre_1
                    omega_old_seq = np.concatenate(omega_list[1:][::-1], axis=0).reshape((-1,1))
                    y_old_seq = np.concatenate(self.y_list[:-1][::-1], axis=0).reshape(-1,1)
                    ytilde_list = [(y - self.h_fn(x_hat)) for y, x_hat in zip(self.y_list, self.xhat_list[1:])] # 从t=1到ds+1，ytilde_1 = y_1 - self.h_fn(xhat_1)
                    ytilde_old_seq = np.concatenate(ytilde_list[:-1][::-1], axis=0).reshape((-1,1))
                    ytilde_seq = np.concatenate(ytilde_list[1:][::-1], axis=0).reshape((-1,1))
                    ytilde = ytilde_list[-1]
                    yaug_old_seq = np.concatenate((y_old_seq, ytilde_old_seq), axis=0) # yaug for y_augment
                    B = -np.eye(ds)
                    # B = np.concatenate((np.zeros((ds,ds)), -np.eye(ds)), axis=0)
                    X = np.concatenate((omega_old_seq, ytilde_old_seq), axis=0)
                    # X = np.concatenate((omega_old_seq, yaug_old_seq), axis=0)
                    # omega_cmp = -self.gamma*inv(inv(Q)+self.gamma*B.T@MT_leftinv@self.H@M_rightinv@B)@B.T@MT_leftinv@self.H@M_rightinv@self.A@self.M@X
                    omega = -self.gamma*inv(inv(Q)+self.gamma*H_11)@(H_omega@omega_old_seq[:-ds]+H_y@ytilde_seq)
                    # omega += np.random.multivariate_normal(mean=np.zeros((ds,)), cov=1e2*np.eye(ds)).reshape((-1,1))
                    self.x_hat = self.f_fn(x=self.x_hat, u=None) + omega.squeeze()
                    del self.xhat_list[0]
                    self.xhat_list.append(self.x_hat)
                    del self.y_list[0]
                    self.y_list.append(y)
                    #region 计算递归最小二乘(全部H)
            #         A = vectorize(X@X.T).reshape((-1,1))
            #         # omega_seq = np.concatenate((omega_old_seq[ds:], omega), axis=0) # 这个错的在M@X的情况下反而效果好？
            #         omega_seq = np.concatenate((omega, omega_old_seq[:-ds]), axis=0)
            #         #region 验证变换后的系统矩阵是否正确(正确)
            #         # dm = (ds+do)*ds
            #         # B_prime = np.pad(array=np.eye(ds), pad_width=((0,dm-ds),(0,0)))
            #         # A_prime = np.concatenate((np.zeros((ds,dm)), 
            #         #                           np.pad(array=np.eye((ds-1)*ds), pad_width=((0,0),(0,(do+1)*ds))),
            #         #                           self.C@self.M,
            #         #                           np.pad(array=np.eye((ds-1)*do), pad_width=((0,0),(ds*ds, do)))),
            #         #                           axis = 0)
            #         # X_cmp = A_prime@X + B_prime@omega
            #         #endregion
            #         X = np.concatenate((omega_seq, ytilde_seq), axis=0)
            #         # A -= self.gamma* vectorize(X@X.T).reshape((-1,1))
            #         b = ytilde.reshape((1,-1))@inv(R)@ytilde.reshape((-1,1)) + omega.T@inv(Q)@omega + self.gamma* X.T@self.H@X
            #         P = P - P@A@A.T@P/(1+A.T@P@A)
            #         params = params + P@A@(b - A.T@params)
            #         # 参数收敛则提前结束
            #         params_seq.append(params)
            #         if len(params_seq) > 5:
            #             del params_seq[0]
            #             if isConverge(params_seq, tol=1e-6):
            #                 if t > 100 :
            #                     pass
            #                 break
            # # end for t-seq
            # self.H = vec2mat(params.squeeze())
            #endregion
                    #region 计算递归最小二乘(H的前ds行)
                    omega_old = X[:ds]
                    A = (omega_old@X.T).reshape((-1,1))
                    # if np.linalg.norm(A) > 4000:
                    #     update_flag = False
                    #     break
                    omega_seq = np.concatenate((omega, omega_old_seq[ds:]), axis=0)
                    X = np.concatenate((omega_seq, ytilde_seq), axis=0)
                    # A -= self.gamma* (omega@X.T).reshape((-1,1))
                    b = ytilde.reshape((1,-1))@inv(R)@ytilde.reshape((-1,1)) + omega.T@inv(Q)@omega + self.gamma* omega.T@self.H[:ds]@X
                    P = P - P@A@A.T@P/(1+A.T@P@A)
                    params = params + P@A@(b - A.T@params)
                    # 参数收敛则提前结束
                    params_seq.append(params)
                    if len(params_seq) > 5:
                        del params_seq[0]
                        if isConverge(params_seq, tol=1e-6):
                            break
            # end for t-seq
            if update_flag :
                self.H[:ds] = params.reshape((ds,-1))
            #endregion
                    #region 计算批最小二乘
            #         A = vectorize(X@X.T).reshape((-1,1))
            #         omega_seq = np.concatenate((omega, omega_old_seq[:-ds]), axis=0)
            #         X = np.concatenate((omega_seq, ytilde_seq), axis=0)
            #         A_LS += A@A.T
            #         b = ytilde.reshape((1,-1))@inv(R)@ytilde.reshape((-1,1)) + omega.T@inv(Q)@omega + self.gamma* X.T@self.H@X
            #         b_LS += A*b.item()
            # # end for t-seq
            # result = np.linalg.solve(a=A_LS, b=b_LS)
            # self.H = vec2mat(result.squeeze())
            #endregion
                    #region 计算批最小二乘(H的前ds行，有问题)
            #         omega_old = X[:ds]
            #         A = (omega_old@X.T).reshape((-1,1))
            #         omega_seq = np.concatenate((omega, omega_old_seq[:-ds]), axis=0)
            #         X = np.concatenate((omega_seq, ytilde_seq), axis=0)
            #         A_LS += A@A.T
            #         b = ytilde.reshape((1,-1))@inv(R)@ytilde.reshape((-1,1)) + omega.T@inv(Q)@omega + self.gamma* omega.T@self.H[:ds]@X
            #         b_LS += A*b.item()
            # # end for t-seq
            # result = np.linalg.solve(a=A_LS, b=b_LS)
            # self.H[:ds] = result.x.reshape((ds,-1))
            #endregion
            H_abserror_batch.append(np.linalg.norm(self.H[:ds] - H_optim[:ds], ord='fro'))
            #region 绘图
            # plotReward([params[0] for params in params_seq])
            # plotReward([params[1] for params in params_seq])
            # plotReward([params[2] for params in params_seq])
            # plotReward([params[3] for params in params_seq])
            # plotReward([params[4] for params in params_seq])
            # plt.show()
            #endregion
            #region 参数收敛则提前结束
            params_batch.append(params)
            if len(params_batch) > 5:
                del params_batch[0]
                if isConverge(params_batch, tol=1e-6):
                    break
            if i > 5000:
                break
            #endregion
        # end for i-batch
        fig, ax = plt.subplots()
        # plt.yscale('log')
        plt.grid(True)
        ax.plot(range(len(H_abserror_batch)), H_abserror_batch, 'o', color='r', linestyle='--', label="value")
        ax.plot(range(len(H_abserror_batch)), np.average(H_abserror_batch)*np.ones_like(H_abserror_batch), color='r', label="average")
        ax.set_xlim(0, len(H_abserror_batch))
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('||H-P$^{*-1}$||$_F$')
        ax.set_title('(H-P$^{*-1}$)的Frobenius范数')
        ax.legend()
        plt.show()

    def estimate(self, y, Q, R) -> None:
        ds = self.dim_state
        if len(self.y_list) < self.dim_state+1: 
            # 小于nx维的时候，用EKF
            self.x_hat, self.P_hat = EKF(x=self.x_hat, P=self.P_hat, y_next=y, Q=Q, R=R)
            self.xhat_list.append(self.x_hat)
            self.y_list.append(y)
        else :
            H_11 = self.H[0:ds, 0:ds]
            H_omega = self.H[0:ds, ds:ds*ds]
            H_y = self.H[0:ds, ds*ds:]

            # self.xhat_list 从k-nx 到 k+1
            # self.y_list    从k-nx+1 到 k+1
            xpre_list = [self.f_fn(x=x_hat) for x_hat in self.xhat_list[:-1]] # 从t=k-nx+1到k+1，xpre_1 = self.f_fn(xhat_0)
            omega_list = [(x_hat - x_pre) for x_hat, x_pre in zip(self.xhat_list[1:], xpre_list)] # 从t=k-nx到k，omega_0 = xhat_1 - xpre_1
            omega_old_seq = np.concatenate(omega_list[1:][::-1], axis=0).reshape((-1,1))
            y_old_seq = np.concatenate(self.y_list[:-1][::-1], axis=0).reshape(-1,1)
            ytilde_list = [(y - self.h_fn(x_hat)) for y, x_hat in zip(self.y_list, self.xhat_list[1:])] # 从t=1到ds+1，ytilde_1 = y_1 - self.h_fn(xhat_1)
            ytilde_old_seq = np.concatenate(ytilde_list[:-1][::-1], axis=0).reshape((-1,1))
            ytilde_seq = np.concatenate(ytilde_list[1:][::-1], axis=0).reshape((-1,1))
            yaug_old_seq = np.concatenate((y_old_seq, ytilde_old_seq), axis=0) # yaug for y_augment
            B = -np.eye(ds)
            # B = np.concatenate((np.zeros((ds,ds)), -np.eye(ds)), axis=0)
            X = np.concatenate((omega_old_seq, ytilde_old_seq), axis=0)
            # X = np.concatenate((omega_old_seq, yaug_old_seq), axis=0)
            # MT_leftinv = inv(self.M@self.M.T)@self.M
            # M_rightinv = self.M.T@inv(self.M@self.M.T)
            # omega_cmp = -self.gamma*inv(inv(Q)+self.gamma*B.T@MT_leftinv@self.H@M_rightinv@B)@B.T@MT_leftinv@self.H@M_rightinv@self.A@self.M@X
            omega = -self.gamma*inv(inv(Q)+self.gamma*H_11)@(H_omega@omega_old_seq[:-ds]+H_y@ytilde_seq)
            self.x_hat = self.f_fn(x=self.x_hat, u=None) + omega.squeeze()
            del self.xhat_list[0]
            self.xhat_list.append(self.x_hat)
            del self.y_list[0]
            self.y_list.append(y)

    def cal_Hoptim(self, Q, R) -> None:
        ds = self.dim_state
        do = self.dim_obs
        A = dyn.F_real(x=None, u=None)
        A0 = dyn.F(x=None, u=None)
        self.A = A0
        # self.A = np.concatenate((np.concatenate((A,A-A0), axis=0), np.concatenate((np.zeros_like(A0),A0), axis=0)), axis=1)
        C = dyn.H_real(x=None)
        C0 = dyn.H(x=None)
        self.C = C0
        Pinv_optim = cal_Poptim(A=A0, C=C0, Q=Q, R=R, gamma=self.gamma, tol=1e-6)
        T0 = [np.zeros((ds*do,ds)), np.pad(array=-C0, pad_width=((0,(ds-1)*do),(0,0)))]
        A0script = [-np.eye(ds)]
        Ascript = [-np.eye(ds)]
        O = [C]
        O0 = [C0]
        A__nx = A
        A0__nx = A0
        for n in range(ds-1):
            A0script.append(-A0__nx)
            Ascript.append(-A__nx)
            O0.append(C0@A0__nx)
            O.append(C@A__nx)
            T0.append(np.pad(array=-np.stack(arrays=O0[::-1], axis=0).reshape((-1, ds)), pad_width=((0,(ds-2-n)*do),(0,0))))
            A0__nx = A0__nx@A0
            A__nx = A__nx@A
        A0script = np.concatenate(A0script, axis=1)
        Ascript = np.concatenate(Ascript, axis=1)
        O0 = np.concatenate(O0[::-1], axis=0)
        O = np.concatenate(O[::-1], axis=0)
        del T0[-1]
        T0 = np.concatenate(T0, axis=1)
        O0_leftinv = np.linalg.inv(O0.T@O0)@O0.T
        O_leftinv = np.linalg.inv(O.T@O)@O.T
        M_ytilde = A0__nx@O0_leftinv
        M_omega = A0script - M_ytilde@T0

        """no model error"""
        M = np.concatenate((M_omega, M_ytilde), axis=1)
        """with model error"""
        # M_omega = np.concatenate((np.zeros_like(M_omega), M_omega), axis=0)
        # M_y = np.concatenate((A__nx@O_leftinv, (A-A0)@A0script@Ascript[::-1].T@O_leftinv - A0__nx@O0_leftinv@(O-O0)@O_leftinv), axis=0)
        # M_ytilde = np.concatenate((np.zeros_like(M_ytilde), M_ytilde), axis=0)
        # M = np.concatenate((M_omega, M_y, M_ytilde), axis=1)
        # 计算H*
        # Pinv_optim = np.array([[41.72676759, -6.76297993, -0.62842214,  0.13200138],
        #                        [-6.76297993, 28.23041354,  0.37198139, -0.16593805],
        #                        [-0.62842214,  0.37198139,  1.45699344, -0.80860923],
        #                        [ 0.13200138, -0.16593805, -0.80860923, 10.80751611]])
        
        self.M = M
        H = M.T@Pinv_optim@M

        # M_rightinv = self.M.T@inv(self.M@self.M.T)
        # H_cmp = cal_Poptim(A=M_rightinv@A0@self.M, B=M_rightinv, C=C0@self.M, Q=Q, R=R)
        return H

    def reset(self, x0_hat, P0_hat):
        self.x_hat = x0_hat
        self.P_hat = P0_hat
        self.xhat_list = [self.x_hat]
        self.y_list = []


def main():
    #region get data batch
    dataFile = "data/Dynamics2_steps5000_episodes1000_randomSeed0.bin"
    with open(file=dataFile, mode="rb") as f:
        trajs = pickle.load(f)
    x_batch = trajs["x_batch"]
    y_batch = trajs["y_batch"]
    #endregion
    args = def_param2()
    model_paras_dict, _ = set_params(args)
    model = create_model(**model_paras_dict)
    #region define LSTDO estimator, train and test
    # LSTDO_estimator = LSTDO(dim_state=model.dim_state, dim_obs=model.dim_obs, x0_hat=args.x0_hat, f_fn=model.f, h_fn=model.h, gamma=0.9, randSeed=11111)
    # LSTDO_estimator.train(y_batch=y_batch, Q=model.Q, R=model.R, epsilon=1e4)
    # # test
    # dataFile = "data/Dynamics2_steps100_episodes100_randomSeed0.bin"
    # with open(file=dataFile, mode="rb") as f:
    #     trajs = pickle.load(f)
    # x_batch = trajs["x_batch"]
    # y_batch = trajs["y_batch"]
    # simulate(model=model, args=args, agent=LSTDO_estimator, sim_num=50, STATUS="MHE-RLF", x_batch=x_batch, y_batch=y_batch)
    #endregion
    #region define RL_Observer, train and test
    estimator = RL_Observer(dim_state=model.dim_state, dim_obs=model.dim_obs, x0_hat=args.x0_hat, P0_hat=args.P0_hat, f_fn=model.f, h_fn=model.h, gamma=1.0)
    # estimator.H = estimator.cal_Hoptim(Q=3*model.Q, R=3*model.R) # 测试最优H解析表达式是否正确
    estimator.train(y_batch=y_batch, Q=model.Q, R=model.R)
    # test
    dataFile = "data/Dynamics2_steps100_episodes100_randomSeed0.bin"
    with open(file=dataFile, mode="rb") as f:
        trajs = pickle.load(f)
    x_batch = trajs["x_batch"]
    y_batch = trajs["y_batch"]
    simulate(model=model, args=args, agent=estimator, sim_num=50, STATUS="MHE-RLF", x_batch=x_batch, y_batch=y_batch)
    #endregion


if __name__ == "__main__":
    main()