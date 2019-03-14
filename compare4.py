from __future__ import print_function
import numpy as np
import random
import math
import time
from matplotlib import pyplot as plt
from GP_optimizer4 import *
from AG_GP_utils4 import *

def f_AG(x):
    noise=0.5
    D=len(x)
    D1=D>>1
    x_=x[range(0,D1)]
    y_=x[range(D1,D)]
    return f(x_+y_)+np.random.normal(0,noise)

def AG_init_point(dimension,epsilon):
    x0=np.zeros(dimension)
    y0=np.zeros(dimension)
    x0[0]=random.uniform(-0.95,3.2)
    x0[1]=random.uniform(-0.45,4.4)
    for i in range(0,dimension):
        y0[i]=random.uniform(-epsilon,epsilon)
    return x0,y0

def min_through_first_i_indexes(data):
    res=np.zeros(len(data))
    for i in range(1,len(data)+1):
        temp=data[0:i]
        index=np.where(temp==np.min(temp))
        res[i-1]=data[int(index[0])]
    return res

def make_iter_compare(data1,data2,name1="1",name2="2"): ### Question
    p1,=plt.plot(range(0,np.shape(data1)[0]),data1)
    p2,=plt.plot(range(0,np.shape(data2)[0]),data2)
    plt.legend([p1, p2], [name1, name2], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Regret")
    plt.show()

def compare(dimension,epsilon,data_point_num, select_point_num, noise):
    print("##################################################################")
    print("STABLEOPT method")
    time_start=time.time()
    optimizer=STABLEOPT(beta=4*np.ones(data_point_num+select_point_num),init_num=data_point_num,mu0=0,epsilon=epsilon,D=dimension,iter=500,step=[0.05,0.05],lr=[0.1,0.1],noise=noise)
    optimizer.run()
    time_end=time.time()
    print('Time cost of STABLEOPT:',time_end-time_start,"s")
    STABLEOPT_iter_x=optimizer.x[optimizer.init_num:optimizer.T]

    print("##################################################################")
    x0,y0=AG_init_point(dimension,epsilon)
    x0=optimizer.iter_initial_point
    print("AG method")
    time_start=time.time()
    x_opt,AG_iter_x=AG_run(f_AG,x0,y0,step=[0.05,0.05],lr=[0.1,0.1],dis_fun=distance_fun, epsilon=epsilon,iter=select_point_num,inner_iter=1)
    print("Decision:",end="")
    print(x_opt)
    time_end=time.time()
    print('Time cost of AG:',time_end-time_start,"s")

    gt_x=[-0.195,0.284]
    np.savez('data.npz', STABLEOPT_iter_x= STABLEOPT_iter_x, AG_iter_x= AG_iter_x, gt_x= gt_x)

def min_ZOPSGD_multitimes(x,epsilon,times):
    min=1000000
    para=np.linspace(0.5,1.5,times)
    for j in range(0,times):
        temp=f(ZOPSGD_bounded_f(f,x,distance_fun,epsilon,0.1*para[j],x,lr=0.05*para[j],iter=200,Q=10))
        if temp<min:
            min=temp
    return min

def load_data_and_plot(epsilon,select_point_num):
    data=np.load("data.npz")
    STABLEOPT_iter_x=data["STABLEOPT_iter_x"]
    AG_iter_x=data["AG_iter_x"]
    gt_x=data["gt_x"]
    STABLEOPT_iter_y=np.zeros(select_point_num)
    AG_iter_y=np.zeros(select_point_num)
    for i in range(0,select_point_num):
        STABLEOPT_iter_y[i]=min_ZOPSGD_multitimes(STABLEOPT_iter_x[i],epsilon,5)  ### return perturbed x
        AG_iter_y[i]=min_ZOPSGD_multitimes(AG_iter_x[i],epsilon,5) ### return perturbed x
    gt_y=min_ZOPSGD_multitimes(gt_x,epsilon,5) ### return perturbed x
    make_iter_compare(gt_y*np.ones(select_point_num)-STABLEOPT_iter_y,gt_y*np.ones(select_point_num)-AG_iter_y,"STABLEOPT","AG")
    make_iter_compare(min_through_first_i_indexes(gt_y*np.ones(select_point_num)-STABLEOPT_iter_y),min_through_first_i_indexes(gt_y*np.ones(select_point_num)-AG_iter_y),"STABLEOPT","AG")

def recompute_AG(read_filename,load_filename,dimension,epsilon,select_point_num):
    data=np.load(read_filename)
    STABLEOPT_iter_x=data["STABLEOPT_iter_x"]
    x0,y0=AG_init_point(dimension,epsilon)
    x0=STABLEOPT_iter_x[0]
    x_opt,AG_iter_x=AG_run(f_AG,x0,y0,step=[0.1,0.02],lr=[0.05,0.02],dis_fun=distance_fun, epsilon=epsilon,iter=select_point_num,inner_iter=1)
    gt_x=[-0.195,0.284]
    np.savez('data.npz', STABLEOPT_iter_x= STABLEOPT_iter_x, AG_iter_x= AG_iter_x, gt_x= gt_x)

if __name__=="__main__":
    random.seed(212)
    select_point_num=50
    #compare(dimension=2,epsilon=0.5,data_point_num=2,select_point_num=select_point_num,noise=0.5)
    recompute_AG("data0.npz","data.npz",2,0.5,select_point_num)
    load_data_and_plot(epsilon=0.5,select_point_num=select_point_num)

    #x0=[1,1]
    #y0=[0,0]
    #AG_run(f_AG,x0,y0,step=[0.5,0.1],lr=[0.5,0.05*0],dis_fun=distance_fun, epsilon=0.5,iter=100,inner_iter=1)