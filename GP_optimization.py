from __future__ import print_function
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class STABLEOPT:
    def __init__(self,mu_0,sigma_0,sigma,beta,epsilon,D):
        random.seed(10)
        self.mu_0=mu_0
        self.sigma_0=sigma_0
        self.sigma=sigma
        self.t=0
        self.mu_temp=mu_0#mu_{t-1}
        self.sigma_temp=sigma_0#sigma_{t-1}
        self.beta=beta
        self.epsilon=epsilon
        self.T=len(beta)
        self.D=D
        self.x=np.zeros((self.T,self.D))
        self.y=np.zeros(self.T)
        self.results=-10000*np.ones(self.T)
    def k(self,x1,x2):
        l=1
        return np.exp(-np.linalg.norm(x1 - x2)^2/(2*l^2))
    def dis_fun(self,x1,x2):
        return np.linalg.norm(x1 - x2)
    def k_t(self,x):
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k(x,self.x[i])
        return k
    def K_t(self):
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k(self.x[i],self.x[j])
        return K
    def get_value(self,x):
        y=100-np.linalg.norm(x)-np.linalg.norm(x+np.ones(self.D))+np.random.normal(0,self.sigma)#here we can modify y=f(x)
        return y
    def observe(self,x):#add selected data
        self.x[self.t]=x
        self.y[self.t]=self.get_value(x)
        self.t=self.t+1
        return 0
    def init(self):#feed initial data
        x=(random.randint(0, 50)-25)*0.1
        y=(random.randint(0, 50)-25)*0.1
        z=(random.randint(0, 50)-25)*0.1
        self.observe(np.array([x,y,z]));
        return 0
    def run_onestep(self):
        if self.t<self.T:
            def ucb(x):
                mu=((np.array(self.k_t(x)).T).dot(np.linalg.inv(self.K_t()+self.sigma^2*np.identity(t)))).dot(self.y[0:t])
                sigma_square=k(x,x)-((np.array(self.k_t(x)).T).dot(np.linalg.inv(self.K_t()+self.sigma^2*np.identity(t)))).dot(np.array(self.k_t(x)))
                return mu+np.sqrt(self.beta[t]*sigma_square)
            def lcb(x):
                mu=((np.array(self.k_t(x)).T).dot(np.linalg.inv(self.K_t()+self.sigma^2*np.identity(t)))).dot(self.y[0:t])
                sigma_square=k(x,x)-((np.array(self.k_t(x)).T).dot(np.linalg.inv(self.K_t()+self.sigma^2*np.identity(t)))).dot(np.array(self.k_t(x)))
                return mu-np.sqrt(self.beta[t]*sigma_square)
            max=-100000
            for x in range(0,51):#-2.5:0.1:2.5, search in 3 dimensions
                for y in range(0,51):
                    for z in range(0,51):
                       x_t=np.array([x-25,y-25,z-25])
                       x_t=x_t*0.1
                       min=100000
                       x_temp=np.zeros(self.D)
                       for x1 in range(0,2):#
                           for y1 in range(0,2):
                               for z1 in range(0,2):
                                   delta_t=np.array([x1-1,y1-1,z1-1])
                                   delta_t=delta_t*0.1
                                   if self.dis_fun(x_t,x_t+delta_t)<self.epsilon:
                                       value=self.get_value(x+delta_t)
                                       if value<min:
                                           min=value
                                           x_temp=x+delta_t
                       if min>max:
                            max=min
                            self.x[self.t]=x_temp
            self.observe(self.x[self.t])
            self.results[self.t-1]=max#note that t has increased
        else:
            print("t error!")
            return 0
    def run(self):
        init_data=10
        for i in range(0,init_data):
            self.init()
        print("Init done")
        for i in range(0,self.T-init_data):
            print(i+1,end="")
            print("/",end="")
            print(self.T-init_data)
            self.run_onestep()
        index=np.where(self.results==np.max(self.results))
        print("Decision:",end="")
        print(self.x[index])
        print("Error:",end="")
        print(self.dis_fun(self.x[index],0.5*np.ones(self.D)))
        return 0
    def print(self):
        print(self.x)
        print(self.y)
        print(self.results)

if __name__=="__main__":
    optimizer=STABLEOPT(0,2,0.1,4*np.ones(20),0.2,3)
    optimizer.run()
    #optimizer.print()