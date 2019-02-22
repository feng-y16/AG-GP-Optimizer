from __future__ import print_function
import numpy as np
import random
import math
import time
from GP_optimizer import *
from AG_GP_utils import *

def f_AG(x):#将GP论文里面的优化函数转换为适用于AG的目标函数
    D=len(x)
    D1=D>>1
    x_=x[range(0,D1)]
    y_=x[range(D1,D)]
    return f(x_+y_)

def AG_init_point():#AG方法初始点的选取，其中x0就是GP论文里的x，y0是delta
    x0=np.zeros(dimension)
    y0=np.zeros(dimension)
    for i in range(0,dimension):
        x0[i]=random.uniform(-30,30)
        y0[i]=random.uniform(-epsilon,epsilon)
    return x0,y0

if __name__=="__main__":
    random.seed(10)
    dimension=3
    epsilon=0.2

    print("AG method")
    time_start=time.time()
    x0,y0=AG_init_point()
    x_opt,y_opt=AG_maxmin_minbounded(f_AG,x0,y0,step=[1,0.001],lr=[0.5,0.01],epsilon=epsilon,iter=100,inner_iter=2,last_iter=50)
    print("Decision:",end="")
    print(x_opt+y_opt)
    print("Max value=",end="")
    print(f(x_opt+y_opt))
    print("Error:",end="")#这是和全局最大值比较，并没有仔细算rubust最大值，可能有误差
    print(distance_fun(x_opt+y_opt,0.5*np.ones(dimension)))#使用无穷范数衡量
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    print("##################################################################")
    print("STABLEOPT method")
    time_start=time.time()
    optimizer=STABLEOPT(beta=4*np.ones(30),init_num=20,mu0=0,epsilon=epsilon,D=dimension,iter=50,step=[0.3,0.05],lr=[0.05,0.05],init_point_option="best point")
    optimizer.run()
    time_end=time.time()
    print('Time cost of STABLEOPT:',time_end-time_start,"s")