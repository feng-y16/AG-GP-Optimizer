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

def AG_init_point(dimension,epsilon):#AG方法初始点的选取，其中x0就是GP论文里的x，y0是delta
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
        x_opt,y_opt=AG_maxmin_minbounded_f(f_AG,x0,y0,step=[0.5,0.1],lr=[0.05,0.0001],dis_fun=distance_fun, epsilon=epsilon,iter=100,inner_iter=2,last_iter=100)    
        hist[i]=distance_fun(x_opt,[-0.195,0.284])
    plt.hist(hist,100)
    plt.xlabel("error")
    plt.ylabel("times")
    plt.show()

def make_iter_compare(data1,data2,data3,name1="1",name2="2",name3="3"):
    #print(data1)
    #print(data2)
    #print(data3)
    p1,=plt.plot(range(0,np.shape(data1)[0]),data1)
    p2,=plt.plot(range(0,np.shape(data2)[0]),data2)
    p3,=plt.plot(range(0,np.shape(data3)[0]),data3)
    plt.legend([p1, p2, p3], [name1, name2, name3], loc='upper left')
    plt.xlabel("number of iterations")
    plt.ylabel("function value")
    plt.show()

def compare(dimension,epsilon,data_point_num, select_point_num):
    print("##################################################################")
    print("STABLEOPT method")
    time_start=time.time()
    optimizer=STABLEOPT(beta=4*np.ones(data_point_num+select_point_num),init_num=data_point_num,mu0=0,epsilon=epsilon,D=dimension,iter=50,step=[1.2,0.2],lr=[0.03,0.0001],init_point_option="function value weighted")
    optimizer.run()
    time_end=time.time()
    print('Time cost of STABLEOPT:',time_end-time_start,"s")
    STABLEOPT_iter_res=optimizer.iter_res

    print("##################################################################")
    x0,y0=AG_init_point(dimension,epsilon)
    x0=optimizer.iter_initial_point
    #make_hist(100,x0,y0)
    print("AG method")
    time_start=time.time()
    x_opt,y_opt,AG_iter_res=AG_maxmin_minbounded_f(f_AG,x0,y0,step=[0.5,0.1],lr=[0.05,0.005],dis_fun=distance_fun, epsilon=epsilon,iter=select_point_num,inner_iter=1,last_iter=100)
    print("Decision:",end="")
    print(x_opt)
    print("Max value=",end="")
    print(f(x_opt+y_opt))
    print("Error:",end="")
    print(distance_fun(x_opt,[-0.195,0.284]))#使用l2范数衡量
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    gt_value=f(ZOSGD_bounded_f(f,[-0.195,0.284],distance_fun,epsilon,0.1,[-0.195,0.284],lr=0.005,iter=100,Q=10))
    np.savez('data.npz', STABLEOPT_iter_res= STABLEOPT_iter_res, AG_iter_res= AG_iter_res, gt_value= gt_value)

def load_data_and_plot():
    data=np.load("data.npz")
    STABLEOPT_iter_res=data["STABLEOPT_iter_res"]
    AG_iter_res=data["AG_iter_res"]
    gt_value=data["gt_value"]
    make_iter_compare(STABLEOPT_iter_res,AG_iter_res,gt_value*np.ones(len(AG_iter_res)),"STABLEOPT","AG","Global robust value")

if __name__=="__main__":
    random.seed(10)
    compare(dimension=2,epsilon=0.5,data_point_num=20,select_point_num=100)
    #dimension=2
    #epsilon=0.5
    #x0,y0=AG_init_point(dimension,epsilon)
    #x_opt,y_opt,AG_iter_res=AG_maxmin_minbounded_f(f_AG,x0,y0,step=[1.2,0.2],lr=[0.03,0.0001],dis_fun=distance_fun, epsilon=epsilon,iter=100,inner_iter=2,last_iter=100)
    load_data_and_plot()