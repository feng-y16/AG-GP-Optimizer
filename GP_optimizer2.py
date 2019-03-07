from __future__ import print_function
import numpy as np
import random
import math
from AG_GP_utils import ZOSGD_bounded
from AG_GP_utils2 import *

class STABLEOPT:
    def __init__(self,beta,init_num,mu0,epsilon,D,iter=100,step=[0.3,0.05],lr=[0.05,0.05],init_point_option="function value weighted"):
        self.theta0=1
        self.thetai=np.ones(D)
        self.sigma2=1
        self.mu0=mu0
        self.init_num=init_num#初始采样点的数目
        self.iter=iter#maxmin问题的迭代次数
        self.t=0  ##
        self.beta=beta
        self.epsilon=epsilon#delta与x的距离上限
        self.T=len(beta)#beta的长度就是总数据量（=初始采样点数+贝叶斯优化点数） ### SL's question: What is beta?
        self.D=D#维数
        self.x=np.zeros((self.T,self.D))
        self.y=np.zeros(self.T)
        self.results=-10000*np.ones(self.T)#函数值
        self.step=step#对于x和delta的计算梯度的步长
        self.lr=lr#对于x和delta的学习率
        self.iter_initial_point=np.zeros(D)#第一次迭代的起始点
        self.iter_res=np.zeros(self.T-init_num)
        if init_point_option=="best point":#过去最优点开始迭代
            self.init_point_option=1
            return
        if init_point_option=="random sample point":#随机选择过去采样点开始迭代
            self.init_point_option=2
            return
        if init_point_option=="function value weighted":#按照过去采样点函数值加权（线性加权）
            self.init_point_option=3
            return

    def k(self,x1,x2):#核函数 ## SL: checked
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/self.thetai[i]**2
        r=np.sqrt(r)  ## distance function
        return self.theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2) ### ARRD Maten 5/2 kernel
        #return self.theta0**2*math.exp(-r)

    def k2(self,x1,x2,theta0,thetai):#带参数的核函数  ## SL: checked
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/thetai[i]**2
        r=np.sqrt(r)
        return theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2)
        #return self.theta0**2*math.exp(-r)

    def k_t(self,x): ### SL：checked
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k(x,self.x[i])
        return k

    def k_t2(self,x,theta0,thetai): ### SL：checked
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k2(x,self.x[i],theta0,thetai)
        return k

    def K_t(self): ### SL：checked
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k(self.x[i],self.x[j])
        return K

    def K_t2(self,theta0,thetai): ### SL：checked
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k2(self.x[i],self.x[j],theta0,thetai)
        return K

    def get_value(self,x):#优化目标函数，带高斯噪声
        return f(x)  ### SL: checked f

    def observe(self,x):#观测指定点的函数值 ## SL: checked
        self.x[self.t]=x
        self.y[self.t]=self.get_value(x)  ### SL's question： Note that there is a difference between GP's formulation max_x min_delta f(x-\delta), however, our methods solves min_x max_delta  -f(x-\delta), there is sign difference.
        self.t=self.t+1
        return self.y[self.t-1]

    def init(self):#初始添加观测数据
        x=[]
        x.append(random.uniform(-0.95,3.2)) ### SL： single random initial point
        x.append(random.uniform(-0.45,4.4)) ### SL: single random initial point
        self.x[self.t]=np.array(x)
        #if self.t>1:
        #    while not (np.unique(self.x[range(0,self.t+1)])==self.x[range(0,self.t+1)]):
        #        self.x[self.t]=self.x[self.t]+np.random.multivariate_normal(np.zeros(self.D),0.01*np.identity(self.D))
        self.y[self.t]=self.get_value(np.array(x)) ### SL's question: This is not noisy observations
        self.results[self.t]=f(ZOSGD_bounded_f(f,self.x[self.t],distance_fun,self.epsilon,0.1,self.x[self.t],self.lr[1],100)) ### SL's question: Why do you need it.
        self.t=self.t+1
        return 0

    def get_prior(self):#获得先验, hyper-parameter optimization
        m=np.mean(self.y[range(0,self.t)])
        def log_likehood(x): ### maximize the loglikelihood
            theta0=x[0]
            thetai=x[range(1,self.D+1)]
            mu0=x[self.D+1]    ### prior in mu
            sigma2=x[self.D+2] ### prior in variance
            tempmatrix=self.K_t2(theta0,thetai)+sigma2*np.identity(self.t) ### SL's question: it does not seem correct, self.K_t2(theta0,thetai)
            try:
                inv=np.linalg.inv(tempmatrix)
            except:
                print("Singular matrix when computing prior. Small identity added.")
                inv=np.linalg.inv(tempmatrix+0.1*np.identity(self.t))
            finally:
                return -0.5*(((self.y[range(0,self.t)]-m).T).dot(inv)).dot(self.y[range(0,self.t)]-m)-0.5*math.log(abs(np.linalg.det(tempmatrix)+1e-10))-0.5*self.t*math.log(2*math.pi) ### SL's question m is not a constant, m = mu0, and the last term can be ignored
        bound=np.zeros((self.D+3,2))
        for i in range(0,self.D+3):
            bound[i][0]=-1000000
            bound[i][1]=1000000
        bound[self.D+2][0]=0#优化区域。实际上除了sigma平方大于0，无别的限制
        x_opt=ZOSGD_bounded(log_likehood,np.ones(self.D+3),bound,5,1e-2,100) ### SL's equestion, ZO-signSGD update is better
        self.theta0=x_opt[0]
        self.thetai=x_opt[range(1,self.D+1)]
        self.mu0=x_opt[self.D+1]
        self.sigma2=x_opt[self.D+2]
        return 0

    def pred(self,x):#预测指定x处的mu; SL: checked.
        shape_x=np.shape(x)
        mu=np.zeros(shape_x[0])
        for i in range(0,shape_x[0]):
             try:
                inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
             except:
                print("Singular matrix when computing pred. Small identity added.")
                inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
             finally:
                mu[i]=self.mu0+((np.array(self.k_t(x[i])).T).dot(inv)).dot(self.y[0:self.t]-np.mean(self.y[0:self.t]))
                return mu
    
    def get_iter_res(self):
        for i in range(0,self.T-self.init_num):
            temp=self.results[0:i+self.init_num+1]
            index=np.where(temp==np.max(temp))
            self.iter_res[i]=temp[int(index[0])]

    def select_init_point(self): ### SL's question: did not understand it
        if self.init_point_option==1:
            index=np.where(self.results==np.max(self.results))
            return self.x[index][0]
        if self.init_point_option==2:
            index=random.randint(0,self.t-1)
            return self.x[index]
        if self.init_point_option==3:
            results=self.results[range(0,self.t)]
            results=results-np.min(results)
            results=results/np.sum(results)
            p=random.uniform(0,1)
            temp=0
            for i in range(0,self.t):
                temp=temp+results[i]
                if temp>p:
                    return self.x[i]
            print("Function value weighted method error!")

    def run_onestep(self,iter):#寻找给定先验下下一个最优采样点
        if self.t<self.T:
            def ucb(x):
                try:
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
                except:
                    print("Singular matrix when computing UCB. Small identity added.")
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
                finally:
                    mu=self.mu0+((np.array(self.k_t(x)).T).dot(inv).dot(self.y[0:self.t]))
                    sigma_square=self.theta0**2-(np.array(self.k_t(x)).T).dot(inv).dot(np.array(self.k_t(x)))
                    if(sigma_square<0):  ## SL's question, why it is happened.
                        print("UCB error: sigma2=",end="")
                        print(sigma_square)
                        print("Let sigma2=0")
                        sigma_square=0
                    return mu+np.sqrt(self.beta[self.t]*sigma_square)
            def lcb(x):
                try:
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
                except:
                    print("Singular matrix when computing LCB. Small identity added.")
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
                finally:
                    mu=self.mu0+((np.array(self.k_t(x)).T).dot(inv).dot(self.y[0:self.t]))
                    sigma_square=self.theta0**2-(np.array(self.k_t(x)).T).dot(inv).dot(np.array(self.k_t(x)))
                    if(sigma_square<0): ## SL's question, why it is happened.
                        print("LCB error: sigma2=",end="")
                        print(sigma_square)
                        print("Let sigma2=0")
                        sigma_square=0
                return mu-np.sqrt(self.beta[self.t]*sigma_square)
            x=self.select_init_point() ### SL: I did not understand it.
            if self.t==self.init_num:
                self.iter_initial_point=x ### SL: what dose it mean?
            delta=np.zeros(self.D)
            for i in range(0,iter):
                x=ZOSGA(ucb,x,self.step[0],self.lr[0],1)-delta ### SL's question: why -delta? x is the raw point? The function should be ucb(x+\delta) with respect to x
                delta=ZOSGD_bounded_f(ucb,x,distance_fun,self.epsilon,self.step[1],np.zeros(len(x)),self.lr[1],1) ### SL's question: why not directly do projected ZOSGD? Same thing here, the function should be ucb(x+\delta) but w.r.t. \delta
            delta=ZOSGD_bounded_f(lcb,x,distance_fun,self.epsilon,self.step[1],np.zeros(len(x)),self.lr[1],100)
            self.x[self.t]=x+delta
            #while not (np.unique(self.x[range(0,self.t+1)])==self.x[range(0,self.t+1)]):
            #    self.x[self.t]=self.x[self.t]+np.random.multivariate_normal(np.zeros(self.D),0.01*np.identity(self.D))
            print("Selected x+delta=",end="")
            print(x+delta)
            fun_value=f(ZOSGD_bounded_f(f,self.x[self.t],distance_fun,self.epsilon,self.step[1],self.x[self.t],self.lr[1],100))  #### SL's question, I did not understand it. Where is noise
            print("Function value=",end="")
            print(fun_value)
            self.results[self.t-1]=fun_value#这里t在self.observe()已经更新为现有的点的个数，从而需要-1
            self.observe(x+delta)
        else:
            print("t value error!")
            return 0

    def run(self):
        for i in range(0,self.init_num):
            self.init()
        print("Init done")
        for i in range(0,self.T-self.init_num):
            print(i+1,end="")
            print("/",end="")
            print(self.T-self.init_num)#需要通过贝叶斯方法采样的点的数目
            print("Getting prior……")
            #pred=self.pred(self.x[range(0,self.t)])
            self.get_prior()

            print("theta0=",end="")
            print(self.theta0)
            print("thetai=",end="")
            print(self.thetai)
            print("mu0=",end="")
            print(self.mu0)
            print("sigma2=",end="")
            print(self.sigma2)

            print("Get prior done")
            if self.sigma2<=0:
                print("Prior sigma invaild!")
            self.run_onestep(self.iter)
        self.get_iter_res()
        print("Done.")
        print("")
        index=np.where(self.results==np.max(self.results))
        print("Decision:",end="")
        print(self.x[index][0])
        print("Max value=",end="")
        print(self.results[index][0])
        print("Error:",end="")
        print(distance_fun(self.x[index][0],[0.195,0.284]))
        return 0
    def print(self):
        print("##################################################################")
        print("X:")
        print(self.x)
        print("y:")
        print(self.y)
        print("Function values")
        print(self.results)

if __name__=="__main__":
    random.seed(10)
    #print(ZOSGA_bounded(f,[30,30,30],[[-50,50],[-50,50],[-50,50]],1,0.01,200))
    #print(ZOSGA(f,[30,30,30],1,0.01,200))
    optimizer=STABLEOPT(beta=4*np.ones(30),init_num=20,mu0=0,epsilon=0.2,D=3,iter=50,step=[0.3,0.05],lr=[0.05,0.05],init_point_option="best point")
    optimizer.run()
    optimizer.print()
