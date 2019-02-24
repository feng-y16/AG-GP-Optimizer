from __future__ import print_function
import numpy as np
import random
import math
import time
from matplotlib import pyplot as plt
from GP_optimizer2 import *
from AG_GP_utils2 import *



def f_AG(x):#将GP论文里面的优化函数转换为适用于AG的目标函数
    D=len(x)
    D1=D>>1
    x_=x[range(0,D1)]
    y_=x[range(D1,D)]
    return f(x_+y_)

def AG_init_point():#AG方法初始点的选取，其中x0就是GP论文里的x，y0是delta
    x0=np.zeros(dimension)
    y0=np.zeros(dimension)
    x0[0]=random.uniform(-0.95,3.2)
    x0[1]=random.uniform(-0.45,4.4)
    for i in range(0,dimension):
        y0[i]=random.uniform(-epsilon,epsilon)
    return x0,y0

def make_hist(num,x0,y0):
    hist=np.zeros(num)
    for i in range(0,num):
        print(i)
        x_opt,y_opt=AG_maxmin_minbounded_f(f_AG,x0,y0,step=[0.5,0.1],lr=[0.05,0.0001],dis_fun=distance_fun, epsilon=epsilon,iter=100,inner_iter=2,last_iter=50)    
        hist[i]=distance_fun(x_opt,[-0.195,0.284])
    plt.hist(hist,100)
    plt.xlabel("error")
    plt.ylabel("times")
    plt.show()

if __name__=="__main__":
    random.seed(10)
    dimension=2
    epsilon=0.5

    x0,y0=AG_init_point()
    make_hist(100,x0,y0)
    print("AG method")
    time_start=time.time()
    x_opt,y_opt=AG_maxmin_minbounded_f(f_AG,x0,y0,step=[0.5,0.1],lr=[0.05,0.0001],dis_fun=distance_fun, epsilon=epsilon,iter=100,inner_iter=2,last_iter=50)
    print("Decision:",end="")
    print(x_opt)
    print("Max value=",end="")
    print(f(x_opt+y_opt))
    print("Error:",end="")
    print(distance_fun(x_opt,[-0.195,0.284]))#使用无穷范数衡量
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    print("##################################################################")
    print("STABLEOPT method")
    time_start=time.time()
    optimizer=STABLEOPT(beta=4*np.ones(30),init_num=20,mu0=0,epsilon=epsilon,D=dimension,iter=50,step=[0.3,0.05],lr=[0.05,0.05],init_point_option="best point")
    optimizer.run()
    time_end=time.time()
    print('Time cost of STABLEOPT:',time_end-time_start,"s")
