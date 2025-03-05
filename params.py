import argparse
import numpy as np

# 解析输入参数
def parseParams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", type=str, default="1e-4") # 希望它长什么样就输入什么就行
    parser.add_argument("--goodInit", type=bool, default=False) # ""或不指定表示False，其他都是True
    parser.add_argument("--gamma", type=float, default=0.4)
    args = parser.parse_args()
    return args

# 仅用于生成数据
def getModelParams(modelName):
    if modelName == "Dynamics1":
        modelParams = {
            "x0_mu": np.array([0, 0, 0, 0, 0]),
            "P0": np.diag((1., 1., 1., 1., 1.)),
            "Q": np.array([[0.025/3,0.25/2, 0, 0, 0],[0.25/2, 2.5, 0, 0, 0], [0, 0, 0.025/3,0.25/2, 0],[0, 0, 0.25/2, 2.5, 0], [0, 0, 0, 0, 2.5]]),
            "R": 10*np.eye(3),
            "disturbMu": None,
            "noiseMu": None,
        }
    elif modelName == "Dynamics2" or modelName ==  "Reverse2":
        modelParams = {
            "x0_mu": np.array([10, 10]),
            "P0": np.diag((1., 1.)),
            "Q": np.array([[1,0],[0,1]]),
            "R": np.array([[1]]),
            "disturbMu": None,
            "noiseMu": None,
        }
    elif modelName == "Dynamics3" or modelName ==  "Reverse3":
        modelParams = {
            "x0_mu": np.array([10]),
            "P0": np.array([[1.]]),
            "Q": None,
            "R": None,
            "disturbMu": None,
            "noiseMu": None,
        }
    elif modelName == "Augment2":
        x0_mu = np.array([10,10])
        modelParams = {
            "x0_mu": np.hstack((x0_mu, x0_mu)),
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
    if modelName == "Dynamics1":
        estParams = {
            "x0_hat": np.array([0, 0, 0, 0, 0]),
            "P0_hat": np.diag((1., 1., 1., 1., 1.)),
            "Q": np.array([[0,0, 0, 0, 0],[0, 2.5, 0, 0, 0], [0, 0, 0.025/3,0.25/2, 0],[0, 0, 0.25/2, 2.5, 0], [0, 0, 0, 0, 2.5]]),
            "R": 10*np.eye(3),
        }
    elif modelName == "Dynamics2":
        estParams = {
            "x0_hat": np.array([10, 10]),
            "P0_hat": np.diag((1., 1.)),
            "Q": np.array([[1,0],[0,0]]),
            "R": np.array([[1]]),
        }
    elif modelName == "Reverse2":
        estParams = {
            "x0_hat": np.array([0, 0]),
            "P0_hat": np.diag((10., 10.)),
            "Q": np.array([[0.799935, -0.192448],
                           [1.885995, 0.81918]]),
            "R": np.array([[0.1]]),
        }
    elif modelName == "Dynamics3":
        estParams = {
            "x0_hat": np.array([0]),
            "P0_hat": np.array([[10.]]),
            "Q": np.array([[1.]]),
            "R": np.array([[1.]]),
        }
    elif modelName == "Reverse3":
        estParams = {
            "x0_hat": np.array([10]),
            "P0_hat": np.array([[10.]]),
            "Q": np.array([[1.5625]]),
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
            # "trainEpis": 50,
            "steps": 100,
            "episodes": 100,
            "randSeed": 0,
        }
    trainParams |= args
    return trainParams