import numpy as np
import pickle
import os

from functions import LogFile
from model import getModel
from params import getModelParams

def generate_data(model, modelParam, steps, randSeed):
    np.random.seed(randSeed)
    # 生成初始状态
    if not modelParam["P0"] : 
        initial_state = modelParam["x0_mu"]
    else :
        initial_state = modelParam["x0_mu"] + np.random.multivariate_normal(np.zeros_like(modelParam["x0_mu"]), modelParam["P0"])
    # 生成噪声序列
    if not hasattr(modelParam, "Q") or not modelParam["Q"]:
        disturb_list = np.zeros((steps, model.dim_state))
    else : 
        if not hasattr(modelParam, "disturbMu") or not modelParam["disturbMu"]: 
            modelParam["disturbMu"] = np.zeros(model.dim_state)
        disturb_list = np.random.multivariate_normal(modelParam["disturbMu"], modelParam["Q"], steps)
    if not hasattr(modelParam, "R") or not modelParam["R"]:
        noise_list = np.zeros((steps, model.dim_obs))
    else : 
        if not hasattr(modelParam, "noiseMu") or not modelParam["noiseMu"]: 
            modelParam["noiseMu"] = np.zeros(model.dim_obs)
        noise_list = np.random.multivariate_normal(modelParam["noiseMu"], modelParam["R"], steps)
    # 生成状态序列和观测序列
    t_seq = range(steps)
    x_seq = []
    y_seq = []
    x = initial_state
    for t in t_seq : # 真实轨迹
        x_next, y_next = model.step(x, disturb=disturb_list[t], noise=noise_list[t], isReal=True)
        x = x_next
        x_seq.append(x_next)
        y_seq.append(y_next)
    return x_seq, y_seq

# 生成并保存数据轨迹，对外接口
def generate_trajectories(modelName, steps, episodes, randSeed):
    #region 判断数据是否已经存在
    fileName = f"data/{modelName}_steps{steps}_episodes{episodes}_randomSeed{randSeed}"
    if os.path.isfile(fileName+".bin") :
        print("Data file already exist, input \"y\" to regenerate : ", end="")
        char = input()
        if char != "y" :
            return 
    #endregion
    trajs = {
        "x_batch": [],
        "y_batch": [],
    }
    model = getModel(modelName=modelName)
    modelParam = getModelParams(modelName=modelName)
    for num in range(episodes):
        x_seq, y_seq = generate_data(model=model, modelParam=modelParam, steps=steps, randSeed=randSeed+num)
        trajs["x_batch"].append(x_seq)
        trajs["y_batch"].append(y_seq)

    print("Data saving, please wait...")
    #region save trajectories and relative information
    with open(file=fileName+".bin", mode="wb") as f:
        pickle.dump(trajs, f)
    # relative information
    log = LogFile(fileName=fileName+".txt")
    for key, val in modelParam.items():
        print(f"{key}: {val}")
    log.endLog()
    #endregion
    print("successfully generate data")

if __name__ == "__main__":
    generate_trajectories(modelName="Dynamics3", steps=30, episodes=50, randSeed=10086)
