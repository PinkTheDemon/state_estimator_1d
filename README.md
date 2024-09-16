model.py :
对外：getModel(modelName)——根据模型名称创建并返回对应的模型类

gendata.py :
本文件使用：仅需修改modelName字符串，以及steps、episodes、randSeed参数，即可生成对应数据文件
对外：getData(modelName, steps, episodes, randSeed)——根据参数生成并返回数据轨迹

simulate.py:
本文件使用：
- modelName：通过系统名称指定系统
- steps：单条轨迹仿真步长
- episodes：轨迹总数
- randSeed：随机数种子
- modelErr：模型是否存在误差(有误差就用真实模型估计，无误就用标称模型估计)
- isPrint：是否打印结果（别的方法用EKF初始化的时候不需要打印就可以关掉）
- isPlot：是否绘制结果
- test_options：需要测试的估计方法
- LogFile的参数可以修改结果输出的文件，一般不用改
对外：simulate(agent, estParams, x_batch, y_batch, isPrint=False, isPlot=False)——指定估计器、估计参数、给定数据，进行仿真

linear.py:
本文件使用：python linear.py --cov 1e4 --goodInit 1 --gamma 0.6 --modelErr 1
- modelName：通过系统名称指定测试的系统
- steps：测试单条轨迹步长
- episodes：测试轨迹总数
- randSeed：测试随机数种子
- 训练参数在param.py文件中修改或直接指定

params.py:
对外：get*Params()——可以直接增加非匿名参数，以追加到生成的params字典中。