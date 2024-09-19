import numpy as np

class Model:
    def __init__(self, name, ds, do) -> None:
        self.name = name
        self.dim_state = ds
        self.dim_obs = do
        self.modelErr = False

    # 打印非函数成员变量
    def printAttr(self) -> None:
        model = vars(self)
        for key, val in model.items():
            print(f"{key}: {val}")

    # system dynamics
    def step(self, x, disturb=None, noise=None, isReal=False) : 
        f = self.f_real if isReal else self.f
        h = self.h_real if isReal else self.h
        x_next = f(x=x)
        if disturb is not None : x_next += disturb
        y_next = h(x=x_next)
        if noise is not None : y_next += noise
        return x_next, y_next

    #region 虚函数，子类具体实现，子类没有的话可以直接raise error
    def f(self):
        pass
    def h(self):
        pass
    def F(self):
        pass
    def H(self):
        pass
    # 注意，如果子类不实现real函数，那么在使用real函数时，需要传递关键字参数而非位置参数
    def f_real(self, **args):
        return self.f(**args)
    def h_real(self, **args):
        return self.h(**args)
    def F_real(self, **args):
        return self.F(**args)
    def H_real(self, **args):
        return self.H(**args)
    #endregion

class Dynamics2(Model):
    def __init__(self) -> None:
        super().__init__("Dynamics2", 2, 1)
        self.modelErr = True

    def f(self, x, batch_first:bool=True, **args) : 
        # disceret form
        if batch_first is True : x = x.T
        x_next = np.zeros_like(x)
        x_next[0] = 0.65*x[0] + 0.4*x[1]
        x_next[1] = -0.78*x[0] + 0.74*x[1]
        x = x_next
        if batch_first is True : x = x.T
        return x

    def F(self, **args) : 
        return np.array([[0.65 , 0.40], 
                        [-0.78, 0.74]])
    
    def f_real(self, x, batch_first:bool=True, **args) : 
        # disceret form
        if batch_first is True : x = x.T
        x_next = np.zeros_like(x)
        x_next[0] = 0.95*x[0] + 0.1*x[1]
        x_next[1] = -0.98*x[0] + 0.94*x[1]
        x = x_next
        if batch_first is True : x = x.T
        return x

    def F_real(self, **args) : 
        return np.array([[0.95 , 0.10], 
                        [-0.98, 0.94]])

    def h(self, x, batch_first:bool=True, **args) : 
        if batch_first is True : x = x.T
        y = np.array([x[1]])
        if batch_first is True : y = y.T
        return y

    def H(self, **args) : 
        return np.array([[0,1]])
    
class Dynamics3(Model):
    def __init__(self) -> None:
        super().__init__("Dynamics3", 1, 1)
        self.modelErr = True

    def f(self, x, batch_first:bool=True, **args) : 
        # disceret form
        if batch_first is True : x = x.T
        x_next = np.zeros_like(x)
        x_next[0] = 1.1*x[0]
        x = x_next
        if batch_first is True : x = x.T
        return x

    def F(self, **args) : 
        return np.array([[1.1]])
    
    def f_real(self, x, batch_first:bool=True, **args) : 
        # disceret form
        if batch_first is True : x = x.T
        x_next = np.zeros_like(x)
        x_next[0] = 0.8*x[0]
        x = x_next
        if batch_first is True : x = x.T
        return x

    def F_real(self, **args) : 
        return np.array([[0.8]])

    def h(self, x, batch_first:bool=True, **args) : 
        if batch_first is True : x = x.T
        y = np.array([x[0]])
        if batch_first is True : y = y.T
        return y

    def H(self, **args) : 
        return np.array([[1]])

class Augment2(Model):
    def __init__(self) -> None:
        super().__init__("Augment2", 4, 2)

    def f(self, x, batch_first:bool=True, **args) : 
        # disceret form
        if batch_first is True : x = x.T
        x_next = np.zeros_like(x)
        x_next = np.zeros_like(x)
        x_next[0] = 0.95*x[0] + 0.1*x[1]
        x_next[1] = -0.98*x[0] + 0.94*x[1]
        x_next[2] = 0.3*x[0] - 0.3*x[1] + 0.65*x[2] + 0.4*x[3]
        x_next[3] = -0.2*x[0] + 0.2*x[1] - 0.78*x[2] + 0.74*x[3]
        x = x_next
        if batch_first is True : x = x.T
        return x

    def F(self, **args) : 
        return np.array([[0.95, 0.1, 0, 0],
                        [-0.98, 0.94, 0, 0],
                        [0.3, -0.3, 0.65 , 0.40], 
                        [-0.2, 0.2, -0.78, 0.74]])

    def h(self, x, batch_first:bool=True, **args) : 
        if batch_first is True : x = x.T
        y = np.array([x[1],
                      x[3]])
        if batch_first is True : y = y.T
        return y

    def H(self, **args) : 
        return np.array([[0,1, 0,0],
                         [0,0, 0,1]])

# 对外接口
def getModel(modelName) :
    if modelName == "Dynamics2": return Dynamics2()
    if modelName == "Dynamics3": return Dynamics3()
    if modelName == "Augment2": return Augment2()