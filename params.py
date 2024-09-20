import argparse
import numpy as np

# 解析输入参数
def parseParams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", type=str, default="10000") # 希望它长什么样就输入什么就行
    parser.add_argument("--goodInit", type=bool, default=False) # ""或不指定表示False，其他都是True
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    return args

# 仅用于生成数据
def getModelParams(modelName):
    if modelName == "Dynamics2":
        modelParams = {
            "x0_mu": np.array([10, 10]),
            "P0": np.diag((1., 1.)),
            "Q": None,
            "R": None,
            "disturbMu": None,
            "noiseMu": None,
        }
    elif modelName == "Dynamics3":
        modelParams = {
            "x0_mu": np.array([10]),
            "P0": np.array([[1.]]),
            "Q": None,
            "R": None,
            "disturbMu": None,
            "noiseMu": None,
        }
    elif modelName == "Augment2":
        x0_hat = np.array([0,0])
        x0_mu = np.array([10,10])
        modelParams = {
            "x0_mu": np.hstack((x0_mu, x0_mu-x0_hat)),
            "P0": [[1,0,1,0],
                   [0,1,0,1],
                   [1,0,1,0],
                   [0,1,0,1]], # 只能在modelParam中的P0用list格式，别的只能用ndarray
            "Q": None,
            "R": None,
            "disturbMu": None,
            "noiseMu": None,
        }
    return modelParams

# 状态估计的初始参数
def getEstParams(modelName, **args):
    if modelName == "Dynamics2":
        estParams = {
            "x0_hat": np.array([0, 0]),
            "P0_hat": np.diag((10., 10.)),
            "Q": np.array([[1e0,0],[0,1e0]]),
            "R": np.array([[0.1]]),
        }
    elif modelName == "Dynamics3":
        estParams = {
            "x0_hat": np.array([0]),
            "P0_hat": np.array([[10.]]),
            "Q": np.array([[1.]]),
            "R": np.array([[1.]]),
        }
    elif modelName == "Augment2":
        estParams = {
            "x0_hat": np.array([10,10,10,10]),
            "P0_hat": np.diag((10., 10., 10, 10)),
            "Q": np.array([[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]]),
            "R": np.array([[1,0], [0,1]]),
        }
    estParams |= args
    return estParams

def getTrainParams(estorName, **args):
    if estorName == "RL_Observer":
        trainParams = {
            "trainEpis": 100,
            "steps": 30,
            "episodes": 50,
            "randSeed": 0,
        }
    trainParams |= args
    return trainParams