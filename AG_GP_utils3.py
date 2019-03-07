from __future__ import print_function
import numpy as np
import random

def project(x,bound):
    D=len(x)
    flag=0
    for i in range(0,D):
        if x[i]<bound[i][0]:
            x[i]=bound[i][0]
            flag=1
            continue
        if x[i]>bound[i][1]:
            x[i]=bound[i][1]
            flag=1
            continue
    return x,flag

def project_f_l2(x,x_cen,epsilon):
    D=len(x)
    flag=0
    distance=np.linalg.norm(x-x_cen, ord=2)
    if distance>epsilon:
        flag=1
        x=x_cen+epsilon*(x-x_cen)/distance
    return x,flag

def ZOSIGNSGD_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10):#x0：迭代起始点，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(func(x_opt+u)-func(x_opt))/step
            dx=dx-lr*D*np.sign(grad)*u/Q

        x_temp,flag2=project(x_opt+dx,bound)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
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
                step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGD(func,x0,step,lr=0.1,iter=100,Q=10):#x0：迭代起始点，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(func(x_opt+u)-func(x_opt))/step
            dx=dx-lr*D*grad*u/Q
        x_temp=x_opt+dx
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
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
                step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGA(func,x0,step,lr=0.1,iter=100,Q=10):#x0：迭代起始点，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(func(x_opt+u)-func(x_opt))/step
            dx=dx+lr*D*grad*u/Q
        x_temp=x_opt+dx
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
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
                step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGD_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10):#x0：迭代起始点，bound：每一维的上下界，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(func(x_opt+u)-func(x_opt))/step
            dx=dx-lr*D*grad*u/Q

        x_temp,flag2=project(x_opt+dx,bound)
        if flag2==1:#越界则减小步长和学习率
            lr=lr*0.9
            step=step*0.9
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
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
                step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGA_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10):#x0：迭代起始点，bound：每一维的上下界，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(func(x_opt+u)-func(x_opt))/step
            dx=dx+lr*D*grad*u/Q

        x_temp,flag2=project(x_opt+dx,bound)
        if flag2==1:#越界则减小步长和学习率
            lr=lr*0.9
            step=step*0.9
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
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
                step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGD_bounded_f(func,x0,dis_f,epsilon,step,x_cen,lr=0.1,iter=100,Q=10):#x0：迭代起始点，dis_fun:距离函数，epsilon：离center的最大距离，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(func(x_opt+u)-func(x_opt))/step
            dx=dx-lr*D*grad*u/Q
        x_temp,flag2=project_f_l2(x_opt+dx,x_cen,epsilon)
        if flag2==1:
            lr=lr*0.9#越界则减小步长和学习率
            step=step*0.9
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
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
                step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGA_bounded_f(func,x0,dis_f,epsilon,step,x_cen,lr=0.1,iter=100,Q=10):#x0：迭代起始点，dis_fun:距离函数，epsilon：离center的最大距离，step：计算梯度所用步长，lr：学习率，iter：迭代次数，Q：每次梯度计算采样次数
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm*step
            grad=(func(x_opt+u)-func(x_opt))/step
            dx=dx+lr*D*grad*u/Q
        x_temp,flag2=project_f_l2(x_opt+dx,x_cen,epsilon)
        if flag2==1:
            lr=lr*0.9#越界则减小步长和学习率
            step=step*0.9
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
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
            if flag1%3==0:#多次效果波动，则减小步长和学习率
                step=step*0.95
                lr=lr*0.95
    return x_opt

def AG_maxmin_l2_l2(func,x0,y0,step,lr,dis_fun,epsilon_x,epsilon_y,iter=20,inner_iter=1):#x0：外层max迭代起始点，y0:内层min迭代起始点，step：计算梯度所用步长（x，y各一个），lr：学习率（x，y各一个），dis_fun:距离函数，epsilon_x：x离center的最大距离，epsilon_y：y离center的最大距离，iter：迭代次数，inner_iter：内层迭代次数，last_iter：最后算y的迭代次数
    x_opt=x0
    y_opt=y0
    flag=0
    best_f=-1000000
    AG_iter_res=np.zeros((iter,len(x0)))
    for i in range(0,iter):
        def func_yfixed(x):
            return func(np.hstack((x,y_opt)))
        x_opt=(ZOPSGA_bounded_f(func_yfixed,x_opt,dis_fun,epsilon_x,step[0],x0,lr[0],inner_iter))
        #print("x_opt=",end="")
        #print(x_opt)
        #print("step_x=",end="")
        #print(step[0])
        #print("lr_x=",end="")
        #print(lr[0])
        def func_xfixed(y):
            return func(np.hstack((x_opt,y)))
        y_opt=ZOPSGD_bounded_f(func_xfixed,y_opt,dis_fun,epsilon_y,step[1],np.zeros(len(y0)),lr[1],inner_iter)

        temp_f=func_xfixed(y_opt)
        AG_iter_res[i]=x_opt
        if temp_f>best_f:
            best_f=temp_f
        else:
            flag=flag+1
        if flag%3==0:
            step[0]=step[0]*0.9
            lr[0]=lr[0]*0.9
    return x_opt,AG_iter_res

def AG_maxmin_bounded_l2(func,x0,y0,step,lr,dis_fun,bound_x,epsilon_y,iter=20,inner_iter=1):#x0：外层max迭代起始点，y0:内层min迭代起始点，step：计算梯度所用步长（x，y各一个），lr：学习率（x，y各一个），dis_fun:距离函数，epsilon：y的范数的上界，iter：迭代次数，inner_iter：内层迭代次数，last_iter：最后算y的迭代次数
    x_opt=x0
    y_opt=y0
    flag=0
    best_f=-1000000
    AG_iter_res=np.zeros((iter,len(x0)))
    for i in range(0,iter):
        def func_yfixed(x):
            return func(np.hstack((x,y_opt)))
        AG_iter_res[i]=x_opt
        x_opt=(ZOPSGA_bounded(func_yfixed,x_opt,bound_x,step[0],lr[0],inner_iter))
        #print("x_opt=",end="")
        #print(x_opt)
        #print("step_x=",end="")
        #print(step[0])
        #print("lr_x=",end="")
        #print(lr[0])
        def func_xfixed(y):
            return func(np.hstack((x_opt,y)))
        y_opt=ZOPSGD_bounded_f(func_xfixed,y_opt,dis_fun,epsilon_y,step[1],np.zeros(len(y0)),lr[1],inner_iter)
        temp_f=func_xfixed(y_opt)
        #print(temp_f)
        if temp_f>best_f:
            best_f=temp_f
        else:
            flag=flag+1
        if flag%3==0:
            step[0]=step[0]*0.9
            lr[0]=lr[0]*0.9
    return x_opt,AG_iter_res

def AG_run(func,x0,y0,step,lr,dis_fun,epsilon,iter=20,inner_iter=2):#x0：外层max迭代起始点，y0:内层min迭代起始点，step：计算梯度所用步长（x，y各一个），lr：学习率（x，y各一个），dis_fun:距离函数，epsilon：y的范数的上界，iter：迭代次数，inner_iter：内层迭代次数，last_iter：最后算y的迭代次数
    D_x=len(x0)
    #D_y=len(y0)
    bound_x=np.ones((D_x,2))
    #bound_y=epsilon*np.ones((D_y,2))
    bound_x[0,:]=[-0.95,3.2]
    bound_x[1,:]=[-0.45,4.4]
    #bound_y[:,0]=-bound_y[:,0]
    x_opt,AG_iter_res=AG_maxmin_bounded_l2(func,x0,y0,step,lr,dis_fun,bound_x,epsilon,iter,inner_iter)
    return x_opt,AG_iter_res

def f(x):#优化目标函数
    x_=x[0]
    y_=x[1]
    return -2*x_**6+12.2*x_**5-21.2*x_**4-6.2*x_+6.4*x_**3+4.7*x_**2-y_**6+11*y_**5-43.3*y_**4+10*y_+74.8*y_**3-56.9*y_**2+4.1*x_*y_+0.1*y_**2*x_**2-0.4*y_**2*x_-0.4*x_**2*y_

def distance_fun(x1,x2):
    return np.linalg.norm(x1-x2,ord=2)