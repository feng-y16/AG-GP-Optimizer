from __future__ import print_function
import numpy as np
import random
import math
from GP_optimization import *
from GP_AG_utils import *

def f_AG(x):
    D=len(x)
    D1=D>>1
    x_=x[range(0,D1)]
    y_=x[range(D1,D)]
    return f(x_+y_)

if __name__=="__main__":
    random.seed(10)
    dimension=3
    epsilon=0.2
    #optimizer=STABLEOPT(beta=4*np.ones(30),init_num=20,mu0=0,epsilon=epsilon,D=dimension,iter=50,step=[0.3,0.05],lr=[0.05,0.05],init_point_option="best point")
    #optimizer.run()
    print("##################################################################")
    x0=np.zeros(dimension)
    y0=np.zeros(dimension)
    for i in range(0,dimension):
        x0[i]=random.uniform(-30,30)
        y0[i]=random.uniform(-epsilon,epsilon)
    print(x0)
    print(y0)
    x_opt,y_opt=AG_maxmin_minbounded(f_AG,x0,y0,step=[1,0.001],lr=[0.8,0.001],epsilon=epsilon,iter=100,inner_iter=2,last_iter=50)
    print("Decision:",end="")
    print(x_opt+y_opt)
    print("Max value=",end="")
    print(f(x_opt+y_opt))
    print("Error:",end="")#这是和全局最大值比较，并没有仔细算rubust最大值，可能有误差
    print(distance_fun(x_opt+y_opt,0.5*np.ones(dimension)))