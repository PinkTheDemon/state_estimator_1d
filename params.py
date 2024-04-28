import torch
from collections import namedtuple

#region global parameters, Can Not change in other program files
globalParams = namedtuple("globalParams", ("device"))
GP = globalParams(
    # device, torch.device
    torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)
#endregion

#region model parameters, Can Not change in other program files
modelParams = namedtuple("modelParams", ("dt", "state_dim", "obs_dim", "x0_mu", "P0", "Q", "R"))
MP = modelParams(
    # dt, float
    0.01,
    # stateDim, int
    1,
    # obsDim, int
    1,
    # x0_mu, tensor
    torch.zeros((1,), device=GP.device),
    # P0, tensor matrix
    torch.FloatTensor([[1.0]], device=GP.device),
    # Q, tensor matrix
    torch.FloatTensor([[1.0]], device=GP.device), 
    # R, tensor matrix
    torch.FloatTensor([[1.0]], device=GP.device),
)
#endregion

#region simulate parameters, Can Not change in other program files
simulateParams = namedtuple("simulateParams", ("max_sim_steps", "max_train_steps", "max_episodes"))
SP = simulateParams(
    # max sim steps, int
    200,
    # max train steps, int
    200,
    # max episodes, int
    1000,
)
#endregion

"""test"""