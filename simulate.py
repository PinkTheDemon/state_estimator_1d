import time
import matplotlib.pyplot as plt
import numpy as np

import estimator as est
import functions as fc
from model import getModel
from gendata import getData
from plot import plotTrajectory
from params import getEstParams, getModelParams


def getSysFuns(model, modelErr):
    if modelErr:
        return model.f_real, model.h_real, model.F_real, model.H_real
    else :
        return model.f, model.h, model.F, model.H

def calPinfty(modelName, gamma=0.9):
    model = getModel(modelName=modelName)
    estParams = getEstParams(modelName=modelName)
    x0_hat=estParams["x0_hat"]
    P0_hat=estParams["P0_hat"]
    # P0_hat = np.array([[ 0.77200664,  0.        ,  0.        ,  0.        ,  0.        ],
    #    [ 0.        , 12.41802979,  0.        ,  0.        ,  0.        ],
    #    [ 0.        ,  0.        , 17.02854556,  0.        ,  0.        ],
    #    [ 0.        ,  0.        ,  0.        , 15.94269775,  0.        ],
    #    [ 0.        ,  0.        ,  0.        ,  0.        , 30.15043784]])
    # Poptim_inv = np.array([[ 0.94468358, -0.27164706,  0.        ,  0.        ,  0.        ],
    #                        [-0.27164706,  0.16548627,  0.        ,  0.        ,  0.        ],
    #                        [ 0.        ,  0.        ,  0.94468358, -0.27164706,  0.        ],
    #                        [ 0.        ,  0.        , -0.27164706,  0.16548627,  0.        ],
    #                        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.27223748]])
    Q=estParams["Q"]
    R=estParams["R"]
    A = model.F()
    C = model.H()
    _, y_batch, _ = getData(modelName=modelName, steps=50000, episodes=1, randSeed=11111)

    # 根据Q是否满秩，判断是否需要生成E矩阵
    n = Q.shape[0]
    non_zeros = [i for i in range(n) if np.linalg.norm(Q[i]) != 0]
    r = len(non_zeros)
    E = np.zeros((r,n))
    for i in range(r) : E[i, non_zeros[i]] = 1
    Q = E @ Q @ E.T
    A = E @ A

    P_seq = [fc.inv(P0_hat)]
    # H_abserror_batch = [np.linalg.norm(fc.inv(P_seq[-1]) - Poptim_inv, ord="fro")]
    c_seq = [0]
    xbar_k = x0_hat
    t = 0
    Nb = 100
    for t in range(1,50001) :
        # KF
        P = fc.inv(P_seq[-1])
        P = fc.inv(E.T @ fc.inv(Q + gamma*A @ P @ A.T) @ E + C.T @ fc.inv(R) @ C)
        # Riccati
        # P = Q + A@(P/gamma)@A.T - A@(P/gamma)@C.T@np.linalg.inv(R + C@(P/gamma)@C.T)@C@(P/gamma)@A.T
        c = 0
        # for i in range(50):
        #     y = y_batch[i][t-1]
        #     xbar_kp1 = A @ xbar_k - P @ C.T @ fc.inv(R) @ (C @ A @ xbar_k - y)
        #     c += (y - C @ xbar_kp1).reshape(1,-1) @ fc.inv(R + C @ P @ C.T) @ (y - C @ xbar_kp1) + c_seq[-1]
        # c /= 50
        P_seq.append(fc.inv(P))
        # H_abserror_batch.append(np.linalg.norm(fc.inv(P) - Poptim_inv, ord="fro"))
        c_seq.append(c)
        if (len(P_seq) > 5) :
            del P_seq[0]
            del c_seq[0]
        if fc.isConverge(P_seq, tol=1e-3, ord='fro'):# and fc.isConverge(c_seq, tol=1e-2) :, criterion=np.linalg.norm
            break
    # print(c_seq)
    print("P_* : ", P_seq[-1])
    print("P_*inv : ", fc.inv(P_seq[-1]))
    print("c_* : ", c_seq[-1])
    print("t : ", t-4)
    # fig, ax = plt.subplots()
    # plt.yscale('log')
    # plt.grid(True)
    # ax.plot(range(len(H_abserror_batch)), H_abserror_batch, 'o', color='r', linestyle='--', label="value")
    # ax.plot(range(len(H_abserror_batch)), np.average(H_abserror_batch)*np.ones_like(H_abserror_batch), color='r', label="average")
    # ax.set_xlim(0, len(H_abserror_batch))
    # # ax.set_xlabel('迭代次数')
    # # ax.set_ylabel('||H-P$^{*-1}$||$_F$')
    # # ax.set_title('(H-P$^{*-1}$)的Frobenius范数')
    # ax.legend()
    # plt.show()
    return P_seq[-1]


# 对外接口
def simulate(agent:est.Estimator, estParams, x_batch, y_batch, u_batch, isPrint=False, isPlot=False):
    # 识别参数
    episodes = len(x_batch)
    steps = len(x_batch[0])
    # 变量初始化
    xhat_batch = []
    yhat_batch = []
    Phat_batch = []
    execution_time = 0
    #region 状态估计
    for y_seq, u_seq in zip(y_batch, u_batch):
        # 变量初始化
        xhat_seq = []
        yhat_seq = []
        Phat_seq = []
        agent.reset(x0_hat=estParams["x0_hat"], P0_hat=estParams["P0_hat"])
        timeStart = time.process_time()
        # 单条轨迹估计
        for y, u in zip(y_seq, u_seq):
            agent.estimate(y=y, Q=estParams["Q"], R=estParams["R"], u=u)
            xhat_seq.append(agent.x_hat)
            yhat_seq.append(agent.y_hat)
            Phat_seq.append(agent.P_hat)
        timeEnd = time.process_time()
        execution_time += 1000 * (timeEnd - timeStart) / steps / episodes
        xhat_batch.append(xhat_seq)
        yhat_batch.append(yhat_seq)
        Phat_batch.append(Phat_seq)
        agent.reset(x0_hat=estParams["x0_hat"], P0_hat=estParams["P0_hat"])
    #endregion 状态估计
    #region 保存数据temp
    # import pickle
    # Q=estParams["Q"]
    # n = Q.shape[0]
    # non_zeros = [i for i in range(n) if np.linalg.norm(Q[i]) != 0]
    # r = len(non_zeros)
    # if r < n:
    #     modelFile = "部分模型未知"
    # else :
    #     modelFile = "模型已知"
    # if agent.name == "EKF":
    #     fileName = f"data/{modelFile}/{model.name}_KF"
    #     with open(file=fileName+".bin", mode="wb") as f:
    #         pickle.dump(xhat_batch, f)
    # elif agent.name == "GKF":
    #     fileName = f"data/{modelFile}/{model.name}_RLO"
    #     with open(file=fileName+".bin", mode="wb") as f:
    #         pickle.dump(xhat_batch, f)
    # elif agent.name == "LSTDO":
    #     fileName = f"data/{modelFile}/{agent.model.name}_RLO"
    #     with open(file=fileName+".bin", mode="wb") as f:
    #         pickle.dump(xhat_batch, f)
    #endregion 保存数据
    # 计算性能指标
    MSE_x, RMSE_x = fc.calMSE(x_batch=[x_seq for x_seq in x_batch], xhat_batch=[x_seq for x_seq in xhat_batch])
    MSE_y, RMSE_y = fc.calMSE(x_batch=y_batch, xhat_batch=yhat_batch)
    # 打印
    if isPrint:
        print(f"state MSE of {agent.name}: {MSE_x}, RMSE: {RMSE_x}")
        print(f"observation MSE of {agent.name}: {MSE_y}, RMSE: {RMSE_y}")
        print(f"average cpu time of {agent.name}: {execution_time} ms")
    else :
        return xhat_batch, Phat_batch
    # 绘图
    if isPlot:
        plotTrajectory(x_seq=x_batch[-1], x_hat_seq=xhat_seq, STATUS=agent.name)
        plt.show()

if __name__ == "__main__" : 
    # 选择模型、仿真步数以及轨迹条数
    model = getModel(modelName="Dynamics1")
    steps = 100
    episodes = 50
    randSeed = 10086
    modelErr = False
    isPrint = True
    isPlot = False
    # 选择执行测试的方法
    test_options = ["GKF"] # "EKF", , "UKF", "UKF-MHE", "FIE", "IEKF", "EKF-MHE"
    # 生成数据以及参数
    x_batch, y_batch, u_batch = getData(modelName=model.name, steps=steps, episodes=episodes, randSeed=randSeed)
    modelParams = getModelParams(modelName=model.name)
    estParams = getEstParams(modelName=model.name, modelErr=modelErr)
    # 重定向系统输出以及打印仿真信息
    logfile = fc.LogFile("output/test_results.txt", rename_option=False)
    print("model params:")
    for key, val in modelParams.items():
        print(f"{key}: {val}")
    print("estimator params:")
    for key, val in estParams.items():
        print(f"{key}: {val}")
    print("********************")
    logfile.flush()
    #endregion
    #region 测试
    for status in test_options:
        if "MHE" in status.upper():
            for i in range(1,2):
                estParams["window"] = i
                print("EKF-MHE, window length:", estParams["window"])
                # 生成MHE类
                f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
                agent = est.MHE(f_fn=f, h_fn=h, F_fn=F, H_fn=H, window=estParams["window"])
                simulate(agent=agent, estParams=estParams, x_batch=x_batch, y_batch=y_batch, u_batch=u_batch, isPrint=isPrint, isPlot=isPlot)
                print("********************")
                logfile.flush()
        elif status.upper() == "EKF":
            print(f"{status.upper()}:", flush=True)
            # 生成EKF类
            f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
            agent = est.EKF_class(f_fn=f, h_fn=h, F_fn=F, H_fn=H)
            simulate(agent=agent, estParams=estParams, x_batch=x_batch, y_batch=y_batch, u_batch=u_batch, isPrint=isPrint, isPlot=isPlot)
            print("********************")
            logfile.flush()
        elif status.upper() == "GKF":
            print(f"{status.upper()}:", flush=True)
            # 生成EKF类
            f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
            agent = est.GKF_class(f_fn=f, h_fn=h, F_fn=F, H_fn=H)
            simulate(agent=agent, estParams=estParams, x_batch=x_batch, y_batch=y_batch, u_batch=u_batch, isPrint=isPrint, isPlot=isPlot)
            print("********************")
            logfile.flush()
    logfile.endLog()
    #endregion