import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.linalg import lstsq
from scipy.optimize import least_squares as ls

import params as pm
import simulate as sim
import functions as fc
import estimator as est
from gendata import getData
from plot import plotReward
from model import getModel, Model
from estimator import EKF, cal_Poptim

#region 设置绘图中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 20

class RL_Observer(est.Estimator):
    '''Optimal Observer Design Using Reinforcement Learning and Quadratic Neural Networks'''
    def __init__(self, model:Model, x0_hat=None, P0_hat=None, gamma=0.6) -> None:
        # 不变
        self.model = model
        self.dim_omega = self.model.dim_state
        self.N = self.model.dim_state
        self.gamma = gamma
        self.value_list = []
        # 可变
        super().__init__(name="RL_Observer", x0_hat=x0_hat, P0_hat=P0_hat)
        self.rH = 2*self.dim_omega if self.model.modelErr else self.dim_omega
        Hsize =  self.N*(self.dim_omega+2*self.model.dim_obs) if self.model.modelErr else self.N*(self.dim_omega+self.model.dim_obs)
        self.H = fc.block_diag((5*np.eye(N=self.rH), np.zeros(shape=(Hsize-self.rH, Hsize-self.rH))))

    def reset(self, x0_hat, P0_hat) -> None:
        self.x_hat = x0_hat
        self.y_hat = None
        self.P_hat = P0_hat
        self.y_list = []
        self.ytilde_list = []
        self.omega_list = []
        self.X = None
        self.value = 0
        multiper = 1
        for value in self.value_list :
            self.value += multiper*value
            multiper *= self.gamma
        self.value_list = []

    def estimate(self, y, Q, R, enableNoise=False) -> None:
        ds = self.model.dim_state
        du = self.dim_omega
        if len(self.omega_list) < self.N+1: 
            # 小于nx维的时候，用EKF
            x_pre = self.model.f(x=self.x_hat)
            self.x_hat, self.P_hat = EKF(x=self.x_hat, P=self.P_hat, y_next=y, model=self.model, Q=Q, R=R)
            self.y_hat = self.model.h(self.x_hat)
            self.omega_list.append(self.x_hat-x_pre)
            self.y_list.append(y)
            ytilde = y - self.y_hat
            self.ytilde_list.append(ytilde)
        else :
            if self.model.modelErr:
                y_list=self.y_list[:-1]
                y_k=self.y_list[-1]
            else :
                y_list = None
                y_k = None
            if self.X is None :
                self.X = self.getX(omega_list=self.omega_list[1:], ytilde_list=self.ytilde_list[:-1], y_list=y_list)
            noise = None
            if enableNoise:
                noise = self.noiseGen.getRandom(mean=np.zeros((du,)), cov=self.cov*np.eye(ds)).reshape((-1,1))
            omega, self.X = self.getOmegaAndX(X=self.X, N=self.N, ytilde_k=self.ytilde_list[-1], Q=Q, y_k=y_k, noise=noise)
            self.x_hat = self.model.f(x=self.x_hat) + omega.reshape(-1,)
            self.y_hat = self.model.h(x=self.x_hat)
            self.ytilde_list = [y - self.y_hat]
            self.y_list = [y]
            ytilde = (y - self.y_hat).reshape(-1,1)
            self.value_list.append(self.X.T@self.X + omega.T@fc.inv(Q)@omega)

    def calKoptim(self, Q, R) -> None:
        # def calAreal(A0, A):
        #     delta = A - A0
        #     ALQR = np.vstack((np.zeros_like(A0), A0))
        #     ALQR = np.hstack((np.vstack((A, delta)), ALQR))
        #     CLQR = np.vstack((np.zeros_like(C0), C0))
        #     CLQR = np.hstack((np.vstack((C, deltaC)), CLQR))
        #     Q_LQR = np.array([[0,0],
        #                         [0,10]])
        #     # Q_LQR = fc.block_diag((fc.inv(R), fc.inv(R)))
        #     R_LQR = fc.inv(Q)
        #     BLQR = np.vstack((np.zeros_like(A), -np.eye(A.shape[0])))
        #     Poptim = cal_Poptim(A=ALQR, B=BLQR, C=CLQR, Q=Q_LQR, R=R_LQR, gamma=0.0007, tol=1e-8)
        #     delta = (-self.gamma*fc.inv(R+ self.gamma*BLQR.T@Poptim@BLQR)@BLQR.T@Poptim@ALQR)[:,:ds]
        #     A0 = A0 + delta
        #     return A0
        ds = self.model.dim_state
        A = self.model.F_real()
        A0 = self.model.F()
        deltaA = np.round(A - A0, decimals=4)
        C = self.model.H_real()
        C0 = self.model.H()
        deltaC = C - C0
        # A0_list = [A0]
        # while True:
        #     A0 = calAreal(A=A, A0=A0)
        #     A0_list.append(A0)
        #     if len(A0_list) > 10:
        #         del A0_list[0]
        #     if fc.isConverge(A0_list, tol=1e-6):
        #         break
        W = -np.eye(ds)
        # _表示下标，__表示上标
        A__Nm1, A0__Nm1, U_N, Uhat_N, V_N, Vhat_N, Lambda_N, T_N, P_N, Phi_N = self.getMatrices()
        A__N = A__Nm1@A
        A0__N = A0__Nm1@A0
        V_N_linv = np.linalg.inv(V_N.T@V_N)@V_N.T
        Vhat_N_linv = np.linalg.inv(Vhat_N.T@Vhat_N)@Vhat_N.T
        # 计算M
        if self.model.modelErr:
            Mhat_omega = Uhat_N - A0__N@Vhat_N_linv@T_N
            M_omega = np.zeros_like(Mhat_omega)
            Mhat_y = (P_N - A0__N@Vhat_N_linv@(Lambda_N + Phi_N))@V_N_linv
            M_y = A__N@V_N_linv
            Mhat_ytilde = A0__N@Vhat_N_linv
            M_ytilde = np.zeros_like(Mhat_ytilde)
            M = np.hstack((M_omega, M_y, M_ytilde))
            Mhat = np.hstack((Mhat_omega, Mhat_y, Mhat_ytilde))
            M = np.vstack((M, Mhat))
            # 计算Pinv_optim(模型误差情况下)
            A0 = np.vstack((np.zeros_like(A0), A0))
            A0 = np.hstack((np.vstack((A, deltaA)), A0))
            C0 = np.vstack((np.zeros_like(C0), C0))
            C0 = np.hstack((np.vstack((C, deltaC)), C0))
            Q_LQR = fc.block_diag((fc.inv(R), fc.inv(R)))
            R_LQR = fc.inv(Q)#np.zeros_like(Q)#
            B0 = np.vstack((np.zeros_like(A), -np.eye(A.shape[0])))
            # Poptim = C0.T@Q_LQR@C0
            Poptim = cal_Poptim(A=A0, B=B0, C=C0, Q=Q_LQR, R=R_LQR, gamma=1, tol=1e-8)
            F1 = (-self.gamma*fc.inv(R_LQR+ self.gamma*B0.T@Poptim@B0)@B0.T@Poptim@A0)[:1]
            Poptim = cal_Poptim(A=A0, B=B0, C=C0, Q=Q_LQR, R=R_LQR, gamma=1, tol=1e-8)
            F2 = (-self.gamma*fc.inv(R_LQR+ self.gamma*B0.T@Poptim@B0)@B0.T@Poptim@A0)[1:]
            F = np.vstack((F1, F2))
            # Poptim = C0.T@Q_LQR@C0
            # F = -0.01*self.gamma*(self.gamma*B0.T@Poptim@B0)@B0.T@Poptim@A0
            F = A0[ds:]
            # B = -np.eye(ds)
            # Poptim = cal_Poptim(A=A, B=B, C=C, Q=fc.inv(R), R=fc.inv(Q), gamma=self.gamma)
            # F = -self.gamma*fc.inv(R+ self.gamma*B.T@Poptim@B)@B.T@Poptim@A
            # F = np.hstack((deltaA, F-deltaA))
        else :
            M_ytilde = A__N@V_N_linv
            M_omega = U_N - M_ytilde@T_N
            M = np.hstack((M_omega, M_ytilde))
            B = -np.eye(ds)
            Poptim = cal_Poptim(A=A, B=B, C=C, Q=fc.inv(R), R=fc.inv(Q), gamma=self.gamma)
            F = -self.gamma*fc.inv(R+ self.gamma*B.T@Poptim@B)@B.T@Poptim@A
        # 计算H*
        self.M = M
        K = F@M
        # 计算A_H矩阵
        do = self.model.dim_obs
        N = self.model.dim_state
        C0 = self.model.H()
        da = N*(ds+2*do) if self.model.modelErr else N*(ds+do)
        A_H = np.zeros((ds, da))
        padWidth = ((0, 0), (0, da - (N-1)*ds))
        A_H = np.vstack((A_H, np.pad(np.eye((N-1)*ds), padWidth)))
        if self.model.modelErr:
            A_H = np.vstack((A_H, np.hstack((C, np.zeros_like(C)))@M))
            padWidth = ((0, 0), (N*ds, (N+1)*do))
            A_H = np.vstack((A_H, np.pad(np.eye((N-1)*do), padWidth)))
        A_H = np.vstack((A_H, np.hstack((deltaC, C0))@M))
        padWidth = ((0, 0), (da-(N)*do, do))
        A_H = np.vstack((A_H, np.pad(np.eye((N-1)*do), padWidth)))
        self.A_H = A_H
        # 计算Kcmp
        H = M.T@Poptim@M
        B = np.pad(np.eye(ds), ((0, da-ds), (0, 0)))
        C = np.hstack((deltaC, C0))@M
        # Hoptim = est.cal_Poptim(A=A_H, B=B, C=C, Q=fc.inv(R), R=fc.inv(Q)) # H本身不满秩，应该无法这样计算H*
        H1 = H[:ds]
        H11 = H1[:, :ds]
        Kcmp = -fc.inv(fc.inv(Q)+H11)@H1@A_H
        return K

    def train(self, x_batch_test, y_batch_test, trainParams, estParams):
        ds = self.model.dim_state
        do = self.model.dim_obs
        _, y_batch = getData(modelName=self.model.name, steps=trainParams["steps"],
                             episodes=trainParams["episodes"], randSeed=trainParams["randSeed"])
        Q = estParams["Q"]
        R = estParams["R"]
        self.cov = trainParams['cov'] if isinstance(trainParams['cov'], np.ndarray) else trainParams['cov']*np.eye(ds)
        self.noiseGen = fc.RandomGenerator(randomFun=np.random.multivariate_normal, rand_num=222)
        print(f"gamma: {self.gamma}, cov: {self.cov}, goodInit:", trainParams["goodInit"])
        # 初始参数
        if trainParams["goodInit"]:
            self.H = self.calHoptim(Q=10*Q, R=10*R)
        Lvec = fc.EVD(self.H).T.reshape((-1,)) # 初始参数
        Lvec_batch = []
        # 对照/基线
        H_optim = self.calHoptim(Q=1*Q, R=1*R)
        H_abserror_batch = [np.linalg.norm(self.H[ds:] - H_optim[ds:], ord='fro')]
        K_optim = self.calK(R=fc.inv(Q), H=H_optim)
        Komega_abserror_batch = [np.linalg.norm(self.calK(R=fc.inv(Q))[0] - K_optim[0], ord='fro')]
        Kytilde_abserror_batch = [np.linalg.norm(self.calK(R=fc.inv(Q))[1] - K_optim[1], ord='fro')]
        Heigvalues_batch = [np.linalg.eigvals(self.H)]
        TDerror_batch = []
        #region 非线性最小二乘相关
        def fun(x, A:List[np.ndarray], b:List[np.ndarray]):
            if isinstance(A, np.ndarray):
                A = fc.block_diag([A for _ in range(self.rH)])
            else :
                A = [fc.block_diag([A_one for _ in range(self.rH)]) for A_one in A]
            f = x.reshape((1,-1))@A@x.reshape((-1,1)) - b
            return f.squeeze()
        def jac(x, A, b):
            if isinstance(A, np.ndarray):
                A = fc.block_diag([A for _ in range(self.rH)])
                J = 2 * A @ x.reshape((-1,1))
            else :
                A = [fc.block_diag([A_one for _ in range(self.rH)]) for A_one in A]
                J = 2*(A@x.reshape((-1,1))).squeeze()
                # J = np.einsum("a,ab->ab", J, Ax) ## 这个应该是多余的
            return J
        #endregion
        # 训练轮数
        for i in range(trainParams["trainEpis"]):
            A, b = self.trainData(y_batch=y_batch, Q=Q, R=R, x0_hat=estParams["x0_hat"], P0_hat=estParams["P0_hat"])
            result = ls(fun=fun, x0=Lvec, method='lm', jac=jac, xtol=1e-8, args=(A, b))#
            TDerror_batch.append(result.cost)
            Lvec = result.x
            L = Lvec.reshape((self.rH,-1))
            self.H = L.T @ L
            H_abserror_batch.append(np.linalg.norm(self.H[ds:] - H_optim[ds:], ord='fro'))
            Komega_abserror_batch.append(np.linalg.norm(self.calK(R=fc.inv(Q))[0] - K_optim[0], ord='fro'))
            Kytilde_abserror_batch.append(np.linalg.norm(self.calK(R=fc.inv(Q))[1] - K_optim[1], ord='fro'))
            Heigvalues_batch.append(np.linalg.eigvals(self.H))
            # 参数收敛则提前结束
            # Lvec_batch.append(Lvec)
            # if len(Lvec_batch) > 10:
            #     del Lvec_batch[0]
            #     if fc.isConverge(Lvec_batch, tol=1e-8):
            #         break
        #region 测试
        xhat_batch_test, _ = sim.simulate(agent=self, estParams=estParams, x_batch=x_batch_test, y_batch=y_batch_test)
        MSE, RMSE = fc.calMSE(x_batch=x_batch_test, xhat_batch=xhat_batch_test)
        print(f"MSE: {MSE}, RMSE: {RMSE}")
        #endregion
        #region 绘图
        fig, axs = plt.subplots(2,3, figsize=(25.6, 14.4), dpi=144)
        axs[0,0].plot(range(len(H_abserror_batch)), H_abserror_batch, 'o', color='r', linestyle='--', label="value")
        axs[0,0].plot(range(len(H_abserror_batch)), np.average(H_abserror_batch)*np.ones_like(H_abserror_batch), color='r', label="average")
        axs[0,0].set_xlim(0, len(H_abserror_batch))
        # axs[0,0].set_xlabel('迭代次数')
        axs[0,0].set_ylabel('||H-P$^{*-1}$||$_F$')
        axs[0,0].set_title('(H-P$^{*-1}$)的Frobenius范数')
        axs[0,0].legend()
        # axs[0,0].yscale('log')
        axs[0,0].grid(True)

        # fig, ax = plt.subplots()
        axs[1,0].plot(range(len(Komega_abserror_batch)), Komega_abserror_batch, 'o', color='r', linestyle='--', label="value")
        axs[1,0].plot(range(len(Komega_abserror_batch)), np.average(Komega_abserror_batch)*np.ones_like(Komega_abserror_batch), color='r', label="average")
        axs[1,0].set_xlim(0, len(Komega_abserror_batch))
        axs[1,0].set_xlabel('迭代次数')
        axs[1,0].set_ylabel('||K$_{\omega}$-K$^{*}_{\omega}$||$_F$')
        axs[1,0].set_title('(K$_{\omega}$-K$^{*}_{\omega}$)的Frobenius范数')
        axs[1,0].legend()
        # axs[1,0].yscale('log')
        axs[1,0].grid(True)

        # fig, ax = plt.subplots()
        axs[1,1].plot(range(len(Kytilde_abserror_batch)), Kytilde_abserror_batch, 'o', color='r', linestyle='--', label="value")
        axs[1,1].plot(range(len(Kytilde_abserror_batch)), np.average(Kytilde_abserror_batch)*np.ones_like(Kytilde_abserror_batch), color='r', label="average")
        axs[1,1].set_xlim(0, len(Kytilde_abserror_batch))
        axs[1,1].set_xlabel('迭代次数')
        axs[1,1].set_ylabel('||K$_{\\tilde{y}}$-K$^{*}_{\\tilde{y}}$||$_F$')
        axs[1,1].set_title('(K$_{\\tilde{y}}$-K$^{*}_{\\tilde{y}}$)的Frobenius范数')
        axs[1,1].legend()
        # axs[1,1].yscale('log')
        axs[1,1].grid(True)

        # fig, ax = plt.subplots()
        colors = ['#800000', '#ff0000', '#ff00ff', '#4b0082', '#8a2be2', '#ee82ee', '#d2691e', '#f5deb3']
        for i in range(ds*(ds+2*do)):# if model_err else range(ds*(ds+do)):
            axs[0,1].plot(range(len(Heigvalues_batch)), [Heigvalues[i] for Heigvalues in Heigvalues_batch], 'o', color=colors[i], linestyle='--', label=f"eigvalue{i+1}")
        axs[0,1].set_xlim(0, len(Heigvalues_batch))
        # axs[0,1].set_xlabel('迭代次数')
        # axs[0,1].set_ylabel('rank(H)')
        axs[0,1].set_title('H矩阵的特征值')
        plt.rcParams['font.size'] = 14
        axs[0,1].legend()
        plt.rcParams['font.size'] = 20
        axs[0,1].set_yscale('log')
        axs[0,1].grid(True)

        axs[0,2].plot(range(len(TDerror_batch)), TDerror_batch, 'o', color='r', linestyle='--', label="value")
        axs[0,2].plot(range(len(TDerror_batch)), np.average(TDerror_batch)*np.ones_like(TDerror_batch), color='r', label="average")
        axs[0,2].set_xlim(0, len(TDerror_batch))
        axs[0,2].set_xlabel('迭代次数')
        axs[0,2].set_ylabel('TDerror')
        axs[0,2].set_title('TD error')
        axs[0,2].legend()
        # axs[0,2].yscale('log')
        axs[0,2].grid(True)
        #endregion
        # 保存训练曲线
        plt.text(x=0.5, y=2.4, s=f"MSE: {MSE}, RMSE: {RMSE}", verticalalignment='top',
                 horizontalalignment='right', transform=plt.gca().transAxes, fontsize=24, color="red")
        initvalue = "好" if trainParams["goodInit"] else "差"
        plt.savefig(f"picture/{self.model.name.lower()}/批LS计算L-初始值较{initvalue}/gamma{self.gamma}_noiseCov{self.cov.diagonal()}.svg", dpi=144, format="svg")

    def getX(self, omega_list, ytilde_list, y_list=None):
        omega_seq = np.concatenate(omega_list[::-1], axis=0).reshape((-1,1))
        ytilde_seq = np.concatenate(ytilde_list[::-1], axis=0).reshape((-1,1))
        if y_list is not None:
            y_seq = np.concatenate(y_list[::-1], axis=0).reshape((-1,1))
            X = np.vstack((omega_seq, y_seq, ytilde_seq))
        else :
            X = np.vstack((omega_seq, ytilde_seq))
        return X

    def getOmegaAndX(self, X, N, ytilde_k, Q, y_k=None, noise=None):
        do = self.model.dim_obs
        du = self.dim_omega
        # 分解
        omega_old_seq = X[:N*du]
        ytilde_old_seq = X[-N*do:]
        if y_k is not None:
            y_old_seq = X[N*du:N*(du+do)]
        # 计算omega_k
        ytilde_seq = np.vstack((ytilde_k.reshape(-1,1), ytilde_old_seq[:-do]))
        yaug_seq = ytilde_seq
        if y_k is not None :
            y_seq = np.vstack((y_k.reshape(-1,1), y_old_seq[:-do]))
            yaug_seq = np.vstack((y_seq, ytilde_seq))
        omega_k = self.K @ X
        if noise is not None:
            omega_k += noise
        # 组合为新的X
        omega_seq = np.vstack((omega_k.reshape(-1,1), omega_old_seq[:-du]))
        X = np.vstack((omega_seq, yaug_seq))
        return omega_k, X

    def getMatrices(self):
        ds = self.model.dim_state
        N = self.N
        A = self.model.F_real(x=None, u=None)
        A0 = self.model.F(x=None, u=None)
        deltaA = A - A0
        C = self.model.H_real(x=None)
        C0 = self.model.H(x=None)
        deltaC = C - C0
        W = -np.eye(ds)
        # k=0
        U_k = np.zeros((ds, N*ds))
        Uhat_k = U_k
        T_k = np.empty((0, U_k.shape[1]))
        V_k = np.empty((0, A.shape[1]))
        Vhat_k = np.empty((0, A0.shape[1]))
        Lambda_k = np.empty((0, A.shape[1]))
        A__k = np.eye(ds)
        A0__k = np.eye(ds)
        P_k = np.zeros_like(A)
        Phi_k = np.empty((0, P_k.shape[1]))
        for k in range(1, N+1): # k=1~N
            T_k = np.vstack((C0@Uhat_k, T_k))
            Phi_k = np.vstack((C0@P_k, Phi_k))
            V_k = np.vstack((C@A__k, V_k))
            Vhat_k = np.vstack((C0@A0__k, Vhat_k))
            Lambda_k = np.vstack((deltaC@A__k, Lambda_k))
            P_k = P_k@A + A0__k@deltaA
            W_k = np.hstack((np.zeros((ds,(N-k)*ds)), W, np.zeros((ds, (k-1)*ds))))
            U_k = A@U_k + W_k
            Uhat_k = A0@Uhat_k + W_k
            # **********
            if k < N :
                A__k = A__k@A
                A0__k = A0__k@A0
        return A__k, A0__k, U_k, Uhat_k, V_k, Vhat_k, Lambda_k, T_k, P_k, Phi_k

    def calK(self, R, H=None):
        # 预处理
        ds = self.model.dim_state
        du = self.model.dim_state
        if H is None :
            H = self.H
        # 计算
        H_11 = H[0:du, 0:du]
        H_omega = H[0:du, du:ds*du]
        H_ytilde = H[0:du, ds*du:]
        K = -self.gamma* fc.inv(R + self.gamma*H_11)
        K_omega =  K @ H_omega
        K_ytilde = K @ H_ytilde
        return (K_omega, K_ytilde)

    def trainData(self, y_batch, Q, R, x0_hat, P0_hat):
        A_batch = []
        b_batch = []
        for y_seq in y_batch:
            A_seq = []
            b_seq = []
            self.reset(x0_hat=x0_hat, P0_hat=P0_hat)
            for y in y_seq:
                self.estimate(y=y, Q=Q, R=R, enableNoise=True)
                if self.X is not None:
                    A_seq.append(self.X@self.X.T)
                    b_seq.append(self.calTDtarget(X=self.X))
            A_batch.extend(A_seq[:-1])
            b_batch.extend(b_seq[1:])
        return A_batch, b_batch

    def calTDtarget(self, X, Q, R):
        do = self.model.dim_obs
        du = self.dim_omega
        N = self.N
        # 取出k时刻数据
        omega_k = X[:du]
        ytilde_k = X[-N*do:-(N-1)*do]
        # 计算TDtarget
        target = omega_k.T@fc.inv(Q)@omega_k + ytilde_k.T@fc.inv(R)@ytilde_k + self.gamma*X.T@self.H@X #
        return target

def main():
    #region 测试的模型和参数
    model = getModel(modelName="Dynamics2")
    # model.modelErr = False
    # model.f = model.f_real
    # model.F = model.F_real
    steps = 100
    episodes = 100
    randSeed = 10086
    #endregion
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
    estParams = pm.getEstParams(modelName=model.name)
    args = pm.parseParams()
    trainParams = pm.getTrainParams(estorName="RL_Observer", cov=eval(args.cov), goodInit=args.goodInit, gamma=args.gamma)
    x_batch_test, y_batch_test = getData(modelName=model.name, steps=steps, episodes=episodes, randSeed=randSeed)
    estimator = RL_Observer(model=model, x0_hat=None, P0_hat=None, gamma=trainParams["gamma"])
    estimator.K = estimator.calKoptim(Q=estParams["Q"], R=estParams["R"]) # 测试最优H解析表达式是否正确
    # estimator.train(x_batch_test=x_batch_test, y_batch_test=y_batch_test, estParams=estParams, trainParams=trainParams)
    # test
    sim.simulate(agent=estimator, estParams=estParams, x_batch=x_batch_test, y_batch=y_batch_test, isPrint=True, isPlot=True)
    #endregion


if __name__ == "__main__":
    main()