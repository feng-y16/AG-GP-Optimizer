from __future__ import print_function
import numpy as np
import random
import math
import time
from matplotlib import pyplot as plt
from GP_optimizer2_review import *
from AG_GP_utils2_review import *



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

def max_first_i_indexes(data):
    res=np.zeros(len(data))
    for i in range(1,len(data)+1):
        temp=data[0:i]
        index=np.where(temp==np.min(temp))
        res[i-1]=data[int(index[0])]
    return res

def make_iter_compare(data1,data2,name1="1",name2="2"):
    #print(data1)
    #print(data2)
    #print(data3)
    y1=max_first_i_indexes(data1)
    y2=max_first_i_indexes(data2)
    p1,=plt.plot(range(0,np.shape(data1)[0]),y1)
    p2,=plt.plot(range(0,np.shape(data2)[0]),y2)
    #p3,=plt.plot(range(0,np.shape(data3)[0]),data3)
    #plt.legend([p1, p2, p3], [name1, name2, name3], loc='upper left')
    plt.legend([p1, p2], [name1, name2], loc='upper left')
    plt.xlabel("Number of iterations")
    plt.ylabel("Regret")
    plt.show()

def compare(dimension,epsilon,data_point_num, select_point_num):
    random.seed(10)

    print("##################################################################")
    print("STABLEOPT method")
    time_start=time.time()
    optimizer=STABLEOPT(beta=4*np.ones(data_point_num+select_point_num),init_num=data_point_num,mu0=0,epsilon=epsilon,D=dimension,iter=50,step=[1.2,0.2],lr=[0.03,0.0001],init_point_option="function value weighted")
    optimizer.run()
    time_end=time.time()
    print('Time cost of STABLEOPT:',time_end-time_start,"s")
    #STABLEOPT_iter_res=optimizer.iter_res
    STABLEOPT_iter_res=optimizer.x[optimizer.init_num:optimizer.T]

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

    gt_value=[-0.195,0.284]
    #gt_value=f(ZOSGD_bounded_f(f,[-0.195,0.284],distance_fun,epsilon,0.1,[-0.195,0.284],lr=0.005,iter=500,Q=10))
    np.savez('data.npz', STABLEOPT_iter_res= STABLEOPT_iter_res, AG_iter_res= AG_iter_res, gt_value= gt_value)

def select_min_ZOSGD_multitimes(x,epsilon,times):
    min=1000000
    para=np.linspace(0.5,1.5,times)
    for j in range(0,times):
        temp=f(ZOSGD_bounded_f(f,x,distance_fun,epsilon,0.1*para[j],x,lr=0.005*para[j],iter=500,Q=10))
        if temp<min:
            min=temp
    return min

def load_data_and_plot(epsilon,select_point_num):
    data=np.load("data.npz")
    STABLEOPT=data["STABLEOPT_iter_res"]
    AG=data["AG_iter_res"]
    gt=data["gt_value"]
    STABLEOPT_iter_res_y=np.zeros(select_point_num)
    AG_iter_res_y=np.zeros(select_point_num)
    for i in range(0,select_point_num):
        STABLEOPT_iter_res_y[i]=select_min_ZOSGD_multitimes(STABLEOPT[i],epsilon,5)
        AG_iter_res_y[i]=select_min_ZOSGD_multitimes(AG[i],epsilon,5)
    gt_value_y=select_min_ZOSGD_multitimes(gt,epsilon,5)
    make_iter_compare(gt_value_y*np.ones(select_point_num)-STABLEOPT_iter_res_y,gt_value_y*np.ones(select_point_num)-AG_iter_res_y,"STABLEOPT","AG")

if __name__=="__main__":
    select_point_num=2
    compare(dimension=2,epsilon=0.5,data_point_num=10,select_point_num=select_point_num)
    #dimension=2
    #epsilon=0.5
    #x0,y0=AG_init_point(dimension,epsilon)
    #x_opt,y_opt,AG_iter_res=AG_maxmin_minbounded_f(f_AG,x0,y0,step=[1.2,0.2],lr=[0.03,0.0001],dis_fun=distance_fun, epsilon=epsilon,iter=100,inner_iter=2,last_iter=100)
    load_data_and_plot(epsilon=0.5,select_point_num=select_point_num)