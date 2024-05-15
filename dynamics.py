import torch
from scipy.integrate import solve_ivp

from params import GP, MP

#region RK45 for solving CDE
def rk45(func, t0, y0, t_end, h, isGrad):
    t = t0
    y = y0.clone().detach().requires_grad_(isGrad)
    while t < t_end:
        k1 = h * func(t, y)
        k2 = h * func(t + 0.25 * h, y + 0.25 * k1)
        k3 = h * func(t + 3/8 * h, y + 3/32 * k1 + 9/32 * k2)
        k4 = h * func(t + 12/13 * h, y + 1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
        k5 = h * func(t + h, y + 439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
        k6 = h * func(t + 0.5 * h, y - 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
        
        y_next = y + 25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 0.2 * k5
        y_error = 1/360 * k1 - 128/4275 * k3 - 2197/75240 * k4 + 1/50 * k5 + 2/55 * k6
        
        t += h
        y = y_next
    return y


# system dynamics
def f_fn(xt :torch.float32, dt :float =None, ut :torch.float32 =None, wt :torch.float32 =None, isGrad :bool =False) -> torch.float32:
    if dt is None : 
        dt = MP.dt
    if ut is None :
        ut = u_fn(xt=xt)
    dx_dt = lambda t, y : (
            torch.FloatTensor([
                -xt**3
            ], device=GP.device)
            )
    xt_p1 = rk45(dx_dt, t0=0, y0=xt, t_end=dt, h=dt, isGrad=isGrad)
    if wt is not None: 
        xt_p1 += wt
    return xt_p1
# dynamics linearization
def F_fn(xt :torch.float32, dt :float =None, ut :torch.float32 =None) -> torch.float32:
    if dt is None : 
        dt = MP.dt
    if ut is None :
        ut = u_fn(xt=xt)
    F = torch.eye(xt.shape[0]) + dt* \
        torch.FloatTensor([
            [-3*xt**2]
        ], device=GP.device)
    return F
# controller
def u_fn(xt :torch.float32) -> torch.float32:
    ut = torch.FloatTensor([
        0
    ], device=GP.device)
    return ut
# observation
def h_fn(xt :torch.float32, vt :torch.float32 =None) -> torch.float32:
    yt = torch.FloatTensor([
            xt**2+xt
        ], device=GP.device)
    if vt is not None:
        yt += vt
    return yt
# observation linearization
def H_fn(xt :torch.float32) -> torch.float32:
    H = torch.FloatTensor([
        [2*xt+1]
    ], device=GP.device)
    return H
