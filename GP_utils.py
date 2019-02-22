from __future__ import print_function
import numpy as np
import random

def ZOSGD(f,x0,step,lr=0.1,iter=100,Q=5):#x0：迭代起始点，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(f(x0+u)-f(x0))/step
            dx=dx-lr*D*grad*u/Q
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
            if flag%2==0:#多次效果波动，则减小步长和学习率
                step=step*0.75
                lr=lr*0.75
            p=2**(-i-1)
            if np.random.uniform()<p:#如果优化结果变差，也有一定概率接受
                x_opt=x_temp
    return x_opt

def ZOSGA(f,x0,step,lr=0.1,iter=100,Q=5):#x0：迭代起始点，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(f(x0+u)-f(x0))/step
            dx=dx-lr*D*grad*u/Q
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
            if flag%2==0:#多次效果波动，则减小步长和学习率
                step=step*0.75
                lr=lr*0.75
            p=2**(-i-1)
            if np.random.uniform()<p:#如果优化结果变差，也有一定概率接受
                x_opt=x_temp
    return x_opt


def ZOSGD_bounded(f,x0,bound,step,lr=0.1,iter=100,Q=5):#x0：迭代起始点，bound：每一维的上下界，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(f(x0+u)-f(x0))/step
            dx=dx-lr*D*grad*u/Q
        flag2=0
        for j in range(0,D):
            if (x_opt[j]+dx[j]<bound[j][0]) or (x_opt[j]+dx[j]>bound[j][1]):
                flag2=1
                break
        if flag2==1:#越界则减小步长和学习率
            lr=lr*0.8
            step=step*0.8
            continue
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            if flag1%3==0:#多次效果波动，则减小步长和学习率
                step=step*0.85
                lr=lr*0.85
            p=2**(-i-1)
            if np.random.uniform()<p:#如果优化结果变差，也有一定概率接受
                x_opt=x_temp
    return x_opt

def ZOSGA_bounded(f,x0,bound,step,lr=0.1,iter=100,Q=5):#x0：迭代起始点，bound：每一维的上下界，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(f(x0+u)-f(x0))/step
            dx=dx-lr*D*grad*u/Q
        flag2=0
        for j in range(0,D):
            if (x_opt[j]+dx[j]<bound[j][0]) or (x_opt[j]+dx[j]>bound[j][1]):
                flag2=1
                break
        if flag2==1:#越界则减小步长和学习率
            lr=lr*0.8
            step=step*0.8
            continue
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            if flag1%3==0:#如果优化结果变差，也有一定概率接受
                step=step*0.85
                lr=lr*0.85
            p=2**(-i-1)
            if np.random.uniform()<p:#如果优化结果变差，也有一定概率接受
                x_opt=x_temp
    return x_opt

def test(x):
    return x[0]**2+x[1]**2