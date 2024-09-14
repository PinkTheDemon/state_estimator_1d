import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import estimator as est
from functions import inv, LogFile, isConverge
from model import getModel
from gendata import getData
from plot import plotTrajectory
from params import getEstParams, getModelParams


def getSysFuns(model, modelErr):
    if modelErr:
        return model.f, model.h, model.F, model.H
    else :
        return model.f_real, model.h_real, model.F_real, model.H_real

def calMSE(x_batch, y_batch, xhat_batch, yhat_batch):
    # state MSE
    x_batch = np.array(x_batch)
    xhat_batch = np.array(xhat_batch)
    SE_x = np.square(x_batch - xhat_batch)
    MSE_x = np.mean(np.mean(SE_x, axis=0), axis=0)
    RMSE_x = np.sqrt(np.mean(MSE_x))
    # observation MSE
    y_batch = np.array(y_batch)
    yhat_batch = np.array(yhat_batch)
    SE_y = np.square(y_batch - yhat_batch)
    MSE_y = np.mean(np.mean(SE_y, axis=0), axis=0)
    RMSE_y = np.sqrt(np.mean(MSE_y))
    return MSE_x, RMSE_x, MSE_y, RMSE_y

# 对外接口
def simulate(agent:est.Estimator, estParams, x_batch, y_batch, isPrint=False, isPlot=False):
    # 识别参数
    episodes = len(x_batch)
    steps = len(x_batch[0])
    # 变量初始化
    xhat_batch = []
    yhat_batch = []
    Phat_batch = []
    execution_time = 0
    #region 状态估计
    for y_seq in y_batch:
        # 变量初始化
        xhat_seq = []
        yhat_seq = []
        Phat_seq = []
        agent.reset(x0_hat=estParams["x0_hat"], P0_hat=estParams["P0_hat"])
        timeStart = time.process_time()
        # 单条轨迹估计
        for y in y_seq:
            agent.estimate(y=y, Q=estParams["Q"], R=estParams["R"])
            xhat_seq.append(agent.x_hat)
            yhat_seq.append(agent.h_fn(x=agent.x_hat))
            Phat_seq.append(agent.P_hat)
        timeEnd = time.process_time()
        execution_time += 1000 * (timeEnd - timeStart) / steps / episodes
        xhat_batch.append(xhat_seq)
        yhat_batch.append(yhat_seq)
        Phat_batch.append(Phat_seq)
    #endregion 状态估计
    # 计算性能指标
    MSE_x, RMSE_x, MSE_y, RMSE_y = calMSE(x_batch=x_batch, y_batch=y_batch, xhat_batch=xhat_batch, yhat_batch=yhat_batch)
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
    model = getModel(modelName="Dynamics3")
    steps = 100
    episodes = 100
    randSeed = 10086
    modelErr = True
    isPrint = True
    isPlot = True
    # 选择执行测试的方法
    test_options = ["EKF"] # , "UKF", "UKF-MHE", "FIE", "IEKF", "EKF-MHE"
    # 生成数据以及参数
    x_batch, y_batch = getData(modelName=model.name, steps=steps, episodes=episodes, randSeed=randSeed)
    modelParams = getModelParams(modelName=model.name)
    estParams = getEstParams(modelName=model.name)
    estParams["modelErr"] = modelErr
    # 重定向系统输出以及打印仿真信息
    logfile = LogFile("output/test_results.txt", rename_option=False)
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
            for i in range(1,10):
                estParams["window"] = i
                print("EKF-MHE, window length:", estParams["window"])
                logfile.flush()
                simulate(model=model, args=None, agent=None, sim_num=50, rand_seed=10086, STATUS=status, x_batch=x_batch, y_batch=y_batch, plot_flag=False)
                print("********************")
        else :
            print(f"{status.upper()}:", flush=True)
            # 生成EKF类
            f, h, F, H = getSysFuns(model=model, modelErr=estParams["modelErr"])
            ekf = est.EKF_class(f_fn=f, h_fn=h, F_fn=F, H_fn=H, dim_state=model.dim_state, dim_obs=model.dim_obs)
            simulate(agent=ekf, estParams=estParams, x_batch=x_batch, y_batch=y_batch, isPrint=isPrint, isPlot=isPlot)
            print("********************")
    logfile.flush()
    logfile.endLog()
    #endregion