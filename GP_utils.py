from __future__ import print_function
import numpy as np
import random

def ZOSGD(f,x0,step,lr=0.1,iter=100):
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag=0
    for i in range(0,iter):
        u = np.random.normal(0, sigma, D)
        u_norm = np.linalg.norm(u)
        u = u / u_norm*step
        grad=(f(x0+u)-f(x0))/step
        dx=-lr*grad*u
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
            if flag%2==0:
                step=step*0.75
                lr=lr*0.75
            p=2**(-i-1)
            if np.random.uniform()<p:
                x_opt=x_temp
    return x_opt

def ZOSGA(f,x0,step,lr=0.1,iter=100):
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag=0
    for i in range(0,iter):
        u = np.random.normal(0, sigma, D)
        u_norm = np.linalg.norm(u)
        u = u / u_norm*step
        grad=(f(x0+u)-f(x0))/step
        dx=lr*grad*u
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
            if flag%2==0:
                step=step*0.75
                lr=lr*0.75
            p=2**(-i-1)
            if np.random.uniform()<p:
                x_opt=x_temp
    return x_opt


def ZOSGD_bounded(f,x0,bound,step,lr=0.1,iter=100):
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        u = np.random.normal(0, sigma, D)
        u_norm = np.linalg.norm(u)
        u = u / u_norm*step
        grad=(f(x0+u)-f(x0))/step
        dx=-lr*grad*u
        flag2=0
        for j in range(0,D):
            if (x_opt[j]+dx[j]<bound[j][0]) or (x_opt[j]+dx[j]>bound[j][1]):
                flag2=1
                break
        if flag2==1:
            lr=lr*0.8
            step=step*0.8
            continue
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            if flag1%3==0:
                step=step*0.95
                lr=lr*0.95
            p=2**(-i-1)
            if np.random.uniform()<p:
                x_opt=x_temp
    return x_opt

def ZOSGA_bounded(f,x0,bound,step,lr=0.1,iter=100):
    D=len(x0)
    x_opt=x0
    best_f=f(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        u = np.random.normal(0, sigma, D)
        u_norm = np.linalg.norm(u)
        u = u / u_norm*step
        grad=(f(x0+u)-f(x0))/step
        dx=lr*grad*u
        flag2=0
        for j in range(0,D):
            if (x_opt[j]+dx[j]<bound[j][0]) or (x_opt[j]+dx[j]>bound[j][1]):
                flag2=1
                break
        if flag2==1:
            lr=lr*0.8
            step=step*0.8
            continue
        x_temp=x_opt+dx
        y_temp=f(x_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
            if flag1%3==0:
                step=step*0.95
                lr=lr*0.95
            p=2**(-i-1)
            if np.random.uniform()<p:
                x_opt=x_temp
    return x_opt

def test(x):
    return x[0]**2+x[1]**2