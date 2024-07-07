import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import estimator as est
from model import create_model
from functions import inv, LogFile, isConverge
from params import def_param2, set_params
from plot import plotTrajectory

dataFile = None
dataFile = "data/Dynamics2_steps100_episodes100_randomSeed10086.bin"

def simulate(model, args, agent=None, sim_num=1, rand_seed=1111, STATUS='EKF', plot_flag=False, x_batch=None, y_batch=None) : 
    #region 常用变量读取
    ds = model.dim_state
    #endregion
    #region 初始化评价指标变量
    MSE_x_avg = 0
    RMSE_x_avg = 0
    MSE_y_avg = 0
    RMSE_y_avg = 0
    execution_time = 0
    count = 0
    #endregion
    #region 估计器模型初始化
    if STATUS.upper() == 'EKF' or STATUS.upper()=='INIT': 
        ekf = est.EKF_class(f_fn=model.f, h_fn=model.h, F_fn=model.F, H_fn=model.H, dim_state=model.dim_state, dim_obs=model.dim_obs)
    elif STATUS == 'PF' : 
        pf = est.Particle_Filter(ds, model.dim_obs, int(1e4), model.f, model.h, model.x0_mu, model.P0)
    #endregion
    #region 生成多条测试轨迹
    for i in range(sim_num) : 
        #region 设置随机数种子
        np.random.seed(rand_seed+i)
        #endregion
        #region 获取一条测试轨迹
        t_seq = range(args.max_sim_steps)
        if x_batch is not None : 
            x_seq = x_batch[i]
            y_seq = y_batch[i]
        else : 
            x_seq, y_seq = model.generate_data(args.max_sim_steps, is_mismatch=args.MODEL_MISMATCH, rand_seed=rand_seed+i)
        #endregion
        #region 记录一条轨迹状态估计所需的cpu时间开始点
        start_time = time.process_time()
        #endregion
        # 状态估计，不同的估计器
        x_hat_seq = []
        P_hat_seq = []
        if STATUS.upper() == 'FIE' : # 全信息估计
            x_hat = args.x0_hat
            P_hat = args.P0_hat
            initial_x = [args.x0_hat]
            # status_seq = [] # debug
            for t in t_seq : 
                result = est.NLSF_uniform(inv(args.P0_hat), y_seq=y_seq[:t+1], Q=model.Q, R=model.R, x0=initial_x, mode="quadratic", x0_bar=args.x0_hat)
                x_hat_seq.append(result.x[-ds:])
                # status_seq.append(result.status) # debug
                initial_x = list(result.x.reshape(-1,ds))[1:]
            # end for t(step)
            # print(f"status seq: {status_seq}") # debug
        elif 'MHE' not in STATUS.upper() : # 单步状态估计，EKF、UKF、PF
            x_hat = args.x0_hat
            P_hat = args.P0_hat
            ekf.reset(x0=x_hat, P0=P_hat)
            for t in t_seq : 
                if STATUS.upper() == 'EKF' or STATUS.upper()=='INIT': 
                    # if t == 24:
                    #     pass
                    # x_next_hat, P_next_hat = est.EKF(x_hat, P_hat, y_seq[t], model.Q, model.R)
                    ekf.estimate(y=y_seq[t], Q=model.Q, R=model.R)
                    x_next_hat = ekf.x_hat
                    P_next_hat = ekf.P_hat
                elif STATUS.upper() == 'IEKF' : 
                    x_next_hat, P_next_hat = est.IEKF(x_hat, P_hat, y_seq[t], model.Q, model.R, times=70)
                elif STATUS.upper() == 'UKF' : 
                    x_next_hat, P_next_hat = est.UKF(x_hat, P_hat, y_seq[t], model.Q, model.R)
                elif STATUS.upper() == 'PF' : 
                    pf.predict(model.Q)
                    pf.update(y_seq[t], model.R)
                    x_next_hat, P_next_hat = pf.estimate()
                # end if STATUS
                x_hat_seq.append(x_next_hat)
                P_hat_seq.append(P_next_hat)
                x_hat = x_next_hat
                P_hat = P_next_hat
            # end for t(step)
        elif 'RLF' in STATUS.upper() and 'MHE' in STATUS.upper() : # RL更新arrival cost的MHE
            agent.reset(x0_hat=args.x0_hat, P0_hat=args.P0_hat)
            # status_seq = [] # debug
            for t in t_seq : 
                agent.estimate(y=y_seq[t], Q=model.Q, R=model.R)
                x_next_hat = agent.x_hat
                x_hat_seq.append(x_next_hat)
                # status_seq.append(agent.sol_status) # debug
            # print(f"status seq: {status_seq}") # debug
        elif 'MHE' in STATUS.upper() : # 传统方法（EKF、UKF）更新arrival cost的MHE
            x0_NLSF = args.x0_hat
            P_hat = args.P0_hat
            initial_x = [args.x0_hat]
            y_list = []
            # status_seq = [] # debug
            for t in t_seq : 
                #region 获取真实观测值
                y_list.append(y_seq[t])
                if len(y_list) > args.window : del y_list[0]
                #endregion
                #region 求解非线性最小二乘，得到 x_next_hat
                result = est.NLSF_uniform(inv(P_hat), y_seq=y_list, Q=model.Q, R=model.R, x0=initial_x, mode="quadratic", x0_bar=x0_NLSF)
                # status_seq.append(result.status) # debug
                x_next_hat = result.x[-ds: ]
                x_hat_seq.append(x_next_hat)
                #endregion
                if t < args.window - 1 : # 窗口未满，x0_NLSF 和 P_hat无需更新
                    initial_x = list(result.x.reshape(-1,ds))
                else : # 窗口已满，用不同的方法更新x0_NLSF 和 P_hat
                    initial_x = list(result.x.reshape(-1,ds))[1:]
                    if 'EKF' in STATUS : 
                        #region 09. Computing arrival cost parameters in moving horizon estimation using sampling based filters
                        # F = model.F(model.f(x0_NLSF))
                        # H = model.H(model.f(x0_NLSF))
                        # P_hat = F@P_hat@F.T + model.Q - F@P_hat@H.T@ inv(H@P_hat@H.T + model.R) @ H@P_hat@F.T
                        #endregion
                        _, P_hat = est.EKF(x0_NLSF, P_hat, y_seq[t-args.window+1], model.Q, model.R)
                    if 'UKF' in STATUS : ## 尚未修改正确
                        _, P_hat = est.UKF(x0_NLSF, P_hat, y_seq[t-args.window+1], model.Q, model.R)
                    x0_NLSF = x_hat_seq[t-args.window+1] # x_hat_k+1|k+1
                # end if t(step)
                P_hat_seq.append(P_hat)
            #end for t(step)
            # print(f"status seq: {status_seq}") # debug
        # end if STATUS
        #region 单步估计所需平均cpu时间，单位ms
        end_time = time.process_time()
        execution_time += 1000 * (end_time - start_time) / args.max_sim_steps / sim_num
        #endregion
        #region 计算MSE指标
        # state MSE
        x_seq = np.array(x_seq)
        x_hat_seq = np.array(x_hat_seq).reshape((-1,ds))
        MSE_x = np.square(x_seq - x_hat_seq).sum(0) / args.max_sim_steps
        RMSE_x = np.sqrt(np.mean(MSE_x))
        MSE_x_avg += MSE_x / sim_num
        RMSE_x_avg += RMSE_x / sim_num
        # observation MSE
        y_seq = np.array(y_seq)
        y_hat_seq = np.array(model.h(x=x_hat_seq, batch_first=True)).reshape((-1,model.dim_obs))
        MSE_y = np.square(y_seq - y_hat_seq).sum(0) / args.max_sim_steps
        RMSE_y = np.sqrt(np.mean(MSE_y))
        MSE_y_avg += MSE_y / sim_num
        RMSE_y_avg += RMSE_y / sim_num
        #endregion
        #region 统计RMSE大于4的数量
        if RMSE_x > 4 : 
            count += 1
        #endregion
        #region 看P是否能收敛
        if "RLF" not in STATUS.upper():
            if isConverge(P_hat_seq[-5:], criterion=np.linalg.norm):
                pass
        #endregion
    # end for i(sim_num)
    # 结果输出或绘制
    if STATUS.upper() == 'INIT' : 
        return x_hat_seq, y_seq, P_hat_seq
    else : 
        #region 打印MSE和时间指标
        print(f"state MSE of {STATUS.upper()}: {MSE_x_avg}, RMSE: {RMSE_x_avg}")
        print(f"observation MSE of {STATUS.upper()}: {MSE_y_avg}, RMSE: {RMSE_y_avg}")
        print(f"average cpu time of {STATUS.upper()}: {execution_time} ms")
        print(f"mismatch number of {STATUS.upper()}: {count}")
        #endregion
        #region 绘图
        if plot_flag : 
            plotTrajectory(x_seq=x_seq, x_hat_seq=x_hat_seq, STATUS=STATUS)
            plt.show()
        #endregion
        return MSE_x_avg, RMSE_x_avg
    # end if STATUS(== INIT)
# end function simulate

if __name__ == "__main__" : 
    #region 选择执行测试的方法
    test_options = ["EKF"] # , "UKF", "UKF-MHE", "FIE", "IEKF", "EKF-MHE"
    #endregion
    #region 重定向系统输出以及打印仿真信息
    logfile = LogFile("output/test_results.txt", rename_option=False)
    args = def_param2()
    model_paras_dict, estimator_paras_dict = set_params(args)
    model = create_model(**model_paras_dict)
    trajs = None
    if dataFile is not None:
        print(dataFile)
        with open(file=dataFile, mode="rb") as f:
            trajs = pickle.load(f)
    x_batch = None if trajs is None else trajs["x_batch"]
    y_batch = None if trajs is None else trajs["y_batch"]
    if y_batch is not None: args.max_sim_steps = len(y_batch[0])
    print(f"sim steps: {args.max_sim_steps}")
    print(f"x0_mu: {model.x0_mu}, x0_hat: {args.x0_hat}")
    print(f"P0_mu: {model.P0}")
    print(f"P0_hat: {args.P0_hat}")
    print(f"Q: {model.Q}")
    # print(f"Q_hat: {args.Q_hat}")
    print(f"R: {model.R}")
    # print(f"R_hat: {args.R_hat}")
    print("********************")
    #endregion
    #region 测试
    for status in test_options:
        if "MHE" in status.upper():
            for i in range(1,10):
                args.window = i
                print(f"EKF-MHE, window length {args.window}: ")
                logfile.flush()
                simulate(model=model, args=args, agent=None, sim_num=50, rand_seed=10086, STATUS=status, x_batch=x_batch, y_batch=y_batch, plot_flag=False)
                print("********************")
        else :
            print(f"{status.upper()}:")
            logfile.flush()
            simulate(model=model, args=args, agent=None, sim_num=50, rand_seed=10086, STATUS=status, x_batch=x_batch, y_batch=y_batch, plot_flag=False)
            print("********************")
    logfile.flush()
    logfile.endLog()
    #endregion