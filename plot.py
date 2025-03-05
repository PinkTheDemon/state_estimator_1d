import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

from functions import checkFilename

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 20

def plotReward(rewardSeq, filename=None) -> None : 
    tSeq = range(len(rewardSeq))
    smoothedReward = [rewardSeq[0]]
    alpha = 1
    for i in range(1, len(rewardSeq)) : 
        # # 指数平滑
        smoothedValue = alpha*rewardSeq[i] + (1-alpha)*smoothedReward[i-1]
        smoothedReward.append(smoothedValue)
        # # ----------
        # 滑动窗口
        # windowLength = 50
        # if i < windowLength : smoothedReward.append(sum(rewardSeq[:i+1])/(i+1))
        # else : smoothedReward.append(sum(rewardSeq[i-windowLength+1:i+1])/windowLength)
        # ----------
    plt.figure()
    plt.plot(tSeq, smoothedReward)
    plt.title("reward curve")
    plt.xlabel("train times")
    plt.ylabel("train reward")
    if filename is not None : 
        filename = checkFilename(filename)
        plt.savefig(filename)


def plotTrajectory(x_seq, x_hat_seq, STATUS="None") -> None:
    x_seq = np.array(x_seq)
    x_hat_seq = np.array(x_hat_seq)
    ds = x_seq[0].size
    max_steps = len(x_seq)
    t_seq = range(max_steps)
    error = np.abs(x_seq - x_hat_seq)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    plt.rcParams['font.size'] = 28
    fig, axs = plt.subplots(ds,1)
    for i in range(ds) : 
        ax = axs[i] if ds > 1 else axs
        ax.plot(t_seq, x_seq.T[i], 'o', label='x_real', color='blue', linestyle='-')
        ax.plot(t_seq, x_hat_seq.T[i], 'o', label='x_hat', color='red', linestyle='-')
        ax.set_xlim(0, max_steps)
        ax.set_ylabel(f'x{i+1}')
    # axs[0].set_title(f'{STATUS}')
    if ds > 1:
        axs[0].set_title(f'{STATUS}状态估计效果图')
        axs[0].legend()
        axs[-1].set_xlabel('时间步')
    else :
        axs.set_title(f'{STATUS}状态估计效果图')
        axs.legend()
        axs.set_xlabel('时间步')
    plt.grid(True)

    fig, ax = plt.subplots()
    color = ['b','g','r','c','m','y','k']
    for i in range(ds) : 
        ax.plot(t_seq, error[:,i], 'o', label=f'x{i+1}', color=color[i], linestyle='--')
        ax.plot(t_seq, np.average(error[:,i])*np.ones_like(t_seq), color=color[i])
    ax.set_xlim(0, max_steps)
    ax.set_xlabel('时间步')
    ax.set_ylabel('绝对值误差')
    # ax.set_title(f'MSE = {MSE}')
    ax.set_title('绝对值误差')
    ax.legend()
    plt.grid(True)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(x_seq[:,0], x_seq[:,1], x_seq[:,2], label='x_real', color='blue')
    # ax.scatter(*x_seq[0], marker='o', color='blue')
    # ax.plot3D(x_hat_seq[:,0], x_hat_seq[:,1], x_hat_seq[:,2], label='x_hat', color='red')
    # ax.scatter(*x_hat_seq[0], marker='o', color='red')
    # ax.set_xlabel('x1')
    # ax.set_ylabel('x2')
    # ax.set_zlabel('x3')
    # ax.legend()
    # ax.set_title('状态轨迹')

def plotH_h(isSave=False, modelKnown=True, modelName="Dynamics1"):
    '''有些参数需要手动调'''
    if modelKnown:
        modelFile = "模型已知"
    else :
        modelFile = "部分模型未知"
    with open(file=f"data/{modelFile}/{modelName}_H_h_10.0.bin", mode="rb") as f:
        H1e1_seq = pickle.load(f)
        h1e1_seq = pickle.load(f)
        d1e1_seq = pickle.load(f)
    with open(file=f"data/{modelFile}/{modelName}_H_h_100.0.bin", mode="rb") as f:
        H1e2_seq = pickle.load(f)
        h1e2_seq = pickle.load(f)
        d1e2_seq = pickle.load(f)
    with open(file=f"data/{modelFile}/{modelName}_H_h_1000.0.bin", mode="rb") as f:
        H1e3_seq = pickle.load(f)
        h1e3_seq = pickle.load(f)
        d1e3_seq = pickle.load(f)
    with open(file=f"data/{modelFile}/{modelName}_H_h_10000.0.bin", mode="rb") as f:
        H1e4_seq = pickle.load(f)
        h1e4_seq = pickle.load(f)
        d1e4_seq = pickle.load(f)
    with open(file=f"data/{modelFile}/{modelName}_H_h_100000.0.bin", mode="rb") as f:
        H1e5_seq = pickle.load(f)
        h1e5_seq = pickle.load(f)
        d1e5_seq = pickle.load(f)

    if modelKnown:
        Poptim_inv = np.array([[ 0.68245832, -0.19916496,  0.        ,  0.        ,  0.        ],
        [-0.19916496,  0.12648837,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.68245832, -0.19916496,  0.        ],
        [ 0.        ,  0.        , -0.19916496,  0.12648837,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.25615528]])
    elif "1" in modelName:
        Poptim_inv = []
    elif "2" in modelName:
        Poptim_inv = np.array([[1.27478034, 1.        ],
                               [1.        , 1.        ]])
    H1e1_seq = [np.linalg.norm(H-Poptim_inv, ord="fro") for H in H1e1_seq]
    H1e2_seq = [np.linalg.norm(H-Poptim_inv, ord="fro") for H in H1e2_seq]
    H1e3_seq = [np.linalg.norm(H-Poptim_inv, ord="fro") for H in H1e3_seq]
    H1e4_seq = [np.linalg.norm(H-Poptim_inv, ord="fro") for H in H1e4_seq]
    H1e5_seq = [np.linalg.norm(H-Poptim_inv, ord="fro") for H in H1e5_seq]
    H1e1_seq = np.array(H1e1_seq)
    H1e2_seq = np.array(H1e2_seq)
    H1e3_seq = np.array(H1e3_seq)
    H1e4_seq = np.array(H1e4_seq)
    H1e5_seq = np.array(H1e5_seq)
    d1e1_seq = np.array(d1e1_seq)
    d1e2_seq = np.array(d1e2_seq)
    d1e3_seq = np.array(d1e3_seq)
    d1e4_seq = np.array(d1e4_seq)
    d1e5_seq = np.array(d1e5_seq)

    # 绘制H-Pinv的F范数
    t_seq = range(len(H1e1_seq))
    fig, ax1 = plt.subplots(figsize=(12,6))#plt.figure(figsize=(12,6))#plt.subplots(1,2, figsize=(12,6), sharey=True)#
    # # Create a GridSpec with two subplots, adjusting the width ratio
    # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    # ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    # 设置每个子图的x轴范围
    ax1.set_xlim(0, 100)
    # ax2.set_xlim(80, 100)
    # 确保x轴只显示整数
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # 绘制图像
    ax1.plot(t_seq, H1e1_seq, 'o', label=r'$\varepsilon=10^1$', color='b', linestyle='-')
    ax1.plot(t_seq, H1e2_seq, 'o', label=r'$\varepsilon=10^2$', color='r', linestyle='-')
    ax1.plot(t_seq, H1e3_seq, 'o', label=r'$\varepsilon=10^3$', color='g', linestyle='-')
    ax1.plot(t_seq, H1e4_seq, 'o', label=r'$\varepsilon=10^4$', color='y', linestyle='-')
    ax1.plot(t_seq, H1e5_seq, 'o', label=r'$\varepsilon=10^5$', color='k', linestyle='-')
    # ax2.plot(t_seq, H1e1_seq, 'o', label=r'$\varepsilon=10^1$', color='b', linestyle='-')
    # ax2.plot(t_seq, H1e2_seq, 'o', label=r'$\varepsilon=10^2$', color='r', linestyle='-')
    # ax2.plot(t_seq, H1e3_seq, 'o', label=r'$\varepsilon=10^3$', color='g', linestyle='-')
    # ax2.plot(t_seq, H1e4_seq, 'o', label=r'$\varepsilon=10^4$', color='y', linestyle='-')
    # ax2.plot(t_seq, H1e5_seq, 'o', label=r'$\varepsilon=10^5$', color='k', linestyle='-')
    ax1.set_yscale('log')
    # ax2.set_yscale('log')
    # 优化坐标显示
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
    # ax2.tick_params(left=False, labelleft=False)
    ax1.grid(True)
    # ax2.grid(True)
    ax1.legend()
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('||H-P$^{*-1}$||$_F$')

    # # 隐藏第二个子图的左边框和第一个子图的右边框
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # # 添加斜线，表示轴是断裂的
    # d = .015  # 斜线的长度
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((1-d, 1+d), (-d, +d), **kwargs)  # 左下到右上
    # ax1.plot((1-d, 1+d),(1-d, 1+d), **kwargs)  # 左上到右下
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1-d, 1+d), **kwargs)  # 右上到左下
    # ax2.plot((-d, +d), (-d, +d), **kwargs)  # 右下到左上
    # plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()

    if isSave:
        plt.savefig(f"picture/{modelFile}/{modelName}_H.eps")

    # 绘制TD-delta
    fig, ax1 = plt.subplots(figsize=(12,6))#plt.figure(figsize=(12,6))#plt.subplots(1,2, figsize=(12,6), sharey=True)#
    # Create a GridSpec with two subplots, adjusting the width ratio
    # gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    # ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    # 设置每个子图的x轴范围
    t_seq = range(1, 1+len(d1e1_seq))
    ax1.set_xlim(1, 100)
    # ax2.set_xlim(80, 100)
    # 确保x轴只显示整数
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # 绘制图像
    ax1.plot(t_seq, d1e1_seq, 'o', label=r'$\varepsilon=10^1$', color='b', linestyle='-')
    ax1.plot(t_seq, d1e2_seq, 'o', label=r'$\varepsilon=10^2$', color='r', linestyle='-')
    ax1.plot(t_seq, d1e3_seq, 'o', label=r'$\varepsilon=10^3$', color='g', linestyle='-')
    ax1.plot(t_seq, d1e4_seq, 'o', label=r'$\varepsilon=10^4$', color='y', linestyle='-')
    ax1.plot(t_seq, d1e5_seq, 'o', label=r'$\varepsilon=10^5$', color='k', linestyle='-')
    # ax2.plot(t_seq, d1e1_seq, 'o', label=r'$\varepsilon=10^1$', color='b', linestyle='-')
    # ax2.plot(t_seq, d1e2_seq, 'o', label=r'$\varepsilon=10^2$', color='r', linestyle='-')
    # ax2.plot(t_seq, d1e3_seq, 'o', label=r'$\varepsilon=10^3$', color='g', linestyle='-')
    # ax2.plot(t_seq, d1e4_seq, 'o', label=r'$\varepsilon=10^4$', color='y', linestyle='-')
    # ax2.plot(t_seq, d1e5_seq, 'o', label=r'$\varepsilon=10^5$', color='k', linestyle='-')
    ax1.set_yscale('log')
    # ax2.set_yscale('log')
    # 优化坐标显示
    # xticks = [i for i in range(41) if (i % 4 == 0)]
    # xticks[0] = 1
    # ax1.set_xticks(xticks)
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
    # ax2.tick_params(left=False, labelleft=False)

    ax1.grid(True)
    # ax2.grid(True)
    ax1.legend()
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel(r'$\delta$')

    # # 隐藏第二个子图的左边框和第一个子图的右边框
    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # # 添加斜线，表示轴是断裂的
    # d = .015  # 斜线的长度
    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((1-d, 1+d), (-d, +d), **kwargs)  # 左下到右上
    # ax1.plot((1-d, 1+d),(1-d, 1+d), **kwargs)  # 左上到右下
    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1-d, 1+d), **kwargs)  # 右上到左下
    # ax2.plot((-d, +d), (-d, +d), **kwargs)  # 右下到左上
    # plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()

    if isSave:
        plt.savefig(f"picture/{modelFile}/{modelName}_delta.eps")

def plot_StateAndAbse(isSave=False, modelKnown=True, modelName="Dynamics1"):
    '''有些参数需要手动调'''
    if modelKnown:
        modelFile = "模型已知"
    else :
        modelFile = "部分模型未知"
    with open(file=f"data/{modelFile}/{modelName}_RLO.bin", mode="rb") as f:
        xhat_batch_RLO = pickle.load(f)
    with open(file=f"data/{modelFile}/{modelName}_KF.bin", mode="rb") as f:
        xhat_batch_KF = pickle.load(f)
    with open(file=f"data/{modelName}_steps100_episodes50_randomSeed10086.bin", mode="rb") as f:
        trajs = pickle.load(f)
        x_batch = trajs["x_batch"]
    xhat_seq_RLO = np.array(xhat_batch_RLO[0])
    xhat_seq_KF = np.array(xhat_batch_KF[0])
    x_seq = np.array(x_batch[0])
    eAbs_seq_RLO = np.abs(xhat_seq_RLO - x_seq)
    eAbs_seq_KF = np.abs(xhat_seq_KF - x_seq)

    if "1" in modelName:
        nFig = 5
        sizeFig = (20,20)
    else :
        nFig = 2
        sizeFig = (20,8)
    t_seq = range(1,len(x_seq)+1)
    fig, axs = plt.subplots(nFig,2, figsize=sizeFig)
    # 确保坐标只显示整数
    # from matplotlib.ticker import MaxNLocator
    # axs[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
    # axs[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))
    # axs[2,1].yaxis.set_major_locator(MaxNLocator(integer=True))
    # axs[3,1].yaxis.set_major_locator(MaxNLocator(integer=True))
    # axs[4,1].yaxis.set_major_locator(MaxNLocator(integer=True))
    # 绘图
    for i in range(nFig):
        axs[i,0].plot(t_seq,        x_seq[:,i], 'o', label='real', color='k', linestyle='-')
        axs[i,0].plot(t_seq, xhat_seq_RLO[:,i], 'o', label='RLO' if modelKnown else 'MU-RLO', color='b', linestyle='-')
        axs[i,0].plot(t_seq,  xhat_seq_KF[:,i], 'o', label='KF' if modelKnown else 'DKF', color='r', linestyle='-')
        axs[i,1].plot(t_seq, eAbs_seq_RLO[:,i], 'o', label='RLO' if modelKnown else 'MU-RLO', color='b', linestyle='-')
        axs[i,1].plot(t_seq,  eAbs_seq_KF[:,i], 'o', label='KF' if modelKnown else 'DKF', color='r', linestyle='-')
        # 坐标轴设置
        axs[i,0].set_xlim(1,100)
        axs[i,1].set_xlim(1,100)
        axs[i,0].tick_params(labelbottom=False)
        axs[i,1].tick_params(labelbottom=False)
        axs[i,0].set_ylabel(rf'$x({i+1})$')
        axs[i,1].set_ylabel(r'$|\tilde{x}$'+f'({i+1})|')
        # 网格线
        axs[i,0].grid(True)
        axs[i,1].grid(True)
    axs[nFig-1,0].tick_params(labelbottom=True)
    axs[nFig-1,1].tick_params(labelbottom=True)
    axs[nFig-1,0].set_xlabel('时间步')
    axs[nFig-1,1].set_xlabel('时间步')

    # 图例
    axs[0,0].legend()
    axs[0,1].legend()

    plt.tight_layout()

    if isSave:
        plt.savefig(f"picture/{modelFile}/{modelName}_State_and_AbsE.eps")

if __name__ == "__main__" : 
    # rewardSeq = [
    # ]

    modelKnown = False
    modelName = "Dynamics2"
    # plotH_h(isSave=True, modelKnown=modelKnown, modelName=modelName)
    plot_StateAndAbse(isSave=True, modelKnown=modelKnown, modelName=modelName)
    plt.show()