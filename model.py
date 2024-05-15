import numpy as np
import torch

from params import GP, MP
from dynamics import f_fn, F_fn, h_fn, H_fn, u_fn

# 分为Model 和 Dynamic i
#region Model class
class Model() : 
    def __init__(self, f_fn=f_fn, F_fn=F_fn, h_fn=h_fn, H_fn=H_fn, u_fn=u_fn) -> None:
        self.state_dim = MP.state_dim
        self.obs_dim = MP.obs_dim
        self.x0_mu = MP.x0_mu
        self.P0 = MP.P0
        self.Q = MP.Q
        self.R = MP.R
        self.f_fn = f_fn
        self.F_fn = F_fn
        self.u_fn = u_fn
        self.h_fn = h_fn
        self.H_fn = H_fn
    # step forward
    def step(self, xt, dt=None, ut=None, wt=None, vt=None) -> torch.float32:
        if dt is None : 
            dt = MP.dt
        if ut is None :
            ut = self.u_fn(xt=xt)
        xt_p1 = self.f_fn(xt=xt, dt=dt, ut=ut, wt=wt)
        yt_p1 = self.h_fn(xt=xt_p1, vt=vt)
        return xt_p1, yt_p1
    # generate data trajectory
    def generate_data(self, maxsteps, w_mu=None, v_mu=None, randSeed=123) -> torch.float32:
        # set random seed
        np.random.seed(randSeed)
        # 生成初始状态
        if self.P0.size == 0 : 
            initial_state = self.x0_mu
        else :
            initial_state = np.random.multivariate_normal(self.x0_mu.cpu().numpy(), self.P0)
            initial_state = torch.FloatTensor(initial_state, device=GP.device)
        # 噪声是否有偏
        if w_mu is None : 
            w_mu = np.zeros(self.Q.shape[0])
        if v_mu is None : 
            v_mu = np.zeros(self.R.shape[0])
        # 生成噪声序列
        if self.Q.size == 0 : 
            w_list = np.zeros((maxsteps, w_mu.size))
        else : 
            w_list = np.random.multivariate_normal(w_mu, self.Q, maxsteps)
        if self.R.size == 0 : 
            v_list = np.zeros((maxsteps, v_mu.size))
        else : 
            v_list = np.random.multivariate_normal(v_mu, self.R, maxsteps)
        w_list = torch.FloatTensor(w_list, device=GP.device)
        v_list = torch.FloatTensor(v_list, device=GP.device)
        # 生成状态序列和观测序列
        t_seq = range(maxsteps)
        x_seq = [initial_state]
        y_seq = []
        for t in t_seq : # 真实轨迹
            x_next, y_next = self.step(xt=x_seq[-1], wt=w_list[t], vt=v_list[t])
            x_seq.append(x_next)
            y_seq.append(y_next)
        x_seq = torch.stack(x_seq)
        y_seq = torch.stack(y_seq)
        return x_seq, y_seq

#region to create a Model class
def create_model() : 
    # 创建模型
    model = Model()

    # 验证模型和给定维度是否一致
    x = torch.ones((MP.state_dim, ))
    try : 
        assert model.f_fn(x).shape == (MP.state_dim, )
        assert model.h_fn(x).shape == (MP.obs_dim, )
    except ValueError : 
        raise ValueError("dynamic or observation functions do not match the model")

    return model
#endregion

#region generate some data trajectories and save them
def generate_trajectories(steps, episodes, randSeed):
    model = Model()
    trajs = {
        "x_batch": [],
        "y_batch": [],
    }
    for num in range(episodes):
        x_seq, y_seq = model.generate_data(maxsteps=steps, randSeed=randSeed+num)
        trajs["x_batch"].append(x_seq)
        trajs["y_batch"].append(y_seq)
    trajs["x_batch"] = torch.stack(trajs["x_batch"])
    trajs["y_batch"] = torch.stack(trajs["y_batch"])

    # #region save trajectories and relative information
    index = 1
    status = 'simulate'
    torch.save(trajs, f"data/{status}Data{index}.bin")
    with open(f"data/{status}Data{index}.txt", "w") as file:
        file.write(MP.__repr__())
#endregion

if __name__ == "__main__":
    generate_trajectories(steps=200, episodes=50, randSeed=10086)