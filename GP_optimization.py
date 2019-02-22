from __future__ import print_function
import numpy as np
import random
import math
from sklearn.gaussian_process.kernels import Matern
from GP_utils import *

class STABLEOPT:
    def __init__(self,beta,init_num,mu0,epsilon,D,iter=100):
        random.seed(10)
        self.theta0=1
        self.thetai=np.ones(D)
        self.sigma2=1
        self.mu0=mu0
        self.init_num=init_num
        self.iter=iter
        self.t=0
        self.beta=beta
        self.epsilon=epsilon
        self.T=len(beta)
        self.D=D
        self.x=np.zeros((self.T,self.D))
        self.y=np.zeros(self.T)
        self.results=-10000*np.ones(self.T)
        self.bound=self.epsilon*np.ones((D,2))
        for i in range(0,D):
            self.bound[i][0]=-self.epsilon

    def k(self,x1,x2):
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/self.thetai[i]**2
        r=np.sqrt(r)
        return self.theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2)
        #return self.theta0**2*math.exp(-r)

    def k2(self,x1,x2,theta0,thetai):
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/thetai[i]**2
        r=np.sqrt(r)
        return theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2)
        #return self.theta0**2*math.exp(-r)

    def dis_fun(self,x1,x2):
        return np.linalg.norm(x1-x2, ord=np.inf)

    def k_t(self,x):
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k(x,self.x[i])
        return k

    def k_t2(self,x,theta0,thetai):
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k2(x,self.x[i],theta0,thetai)
        return k

    def K_t(self):
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k(self.x[i],self.x[j])
        return K

    def K_t2(self,theta0,thetai):
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k2(self.x[i],self.x[j],theta0,thetai)
        return K

    def get_value(self,x):
        y=100.0-np.linalg.norm(x)**2-np.linalg.norm(x-np.ones(self.D))**2+np.random.normal(0,0.1)#here we can modify y=f(x)
        return y

    def observe(self,x):#add selected data
        self.x[self.t]=x
        self.y[self.t]=self.get_value(x)
        self.t=self.t+1
        return self.y[self.t-1]

    def init(self):#feed initial data
        x=random.uniform(-30,30)
        y=random.uniform(-30,30)
        z=random.uniform(-30,30)
        self.x[self.t]=np.array([x,y,z])
        #if self.t>1:
        #    while not (np.unique(self.x[range(0,self.t+1)])==self.x[range(0,self.t+1)]):
        #        self.x[self.t]=self.x[self.t]+np.random.multivariate_normal(np.zeros(self.D),0.01*np.identity(self.D))
        self.y[self.t]=self.get_value(np.array([x,y,z]))
        self.results[self.t]=self.y[self.t]
        self.t=self.t+1
        return 0

    def get_prior(self):
        m=np.mean(self.y[range(0,self.t)])
        def log_likehood(x):
            theta0=x[0]
            thetai=x[range(1,self.D+1)]
            mu0=x[self.D+1]
            sigma2=x[self.D+2]
            tempmatrix=self.K_t2(theta0,thetai)+sigma2*np.identity(self.t)
            try:
                inv=np.linalg.inv(tempmatrix)
            except:
                print("Singular matrix when computing prior. Small identity added.")
                inv=np.linalg.inv(tempmatrix+0.1*np.identity(self.t))
            finally:
                return -0.5*(((self.y[range(0,self.t)]-m).T).dot(inv)).dot(self.y[range(0,self.t)]-m)-0.5*math.log(abs(np.linalg.det(tempmatrix)+1e-10))-0.5*self.t*math.log(2*math.pi)
        bound=np.zeros((self.D+3,2))
        for i in range(0,self.D+3):
            bound[i][0]=-1000000
            bound[i][1]=1000000
        bound[self.D+2][0]=0
        x_opt=ZOSGD_bounded(log_likehood,np.ones(self.D+3),bound,1,0.03,100)
        self.theta0=x_opt[0]
        self.thetai=x_opt[range(1,self.D+1)]
        self.mu0=x_opt[self.D+1]
        self.sigma2=x_opt[self.D+2]
        return 0

    def pred(self,x):
        shape_x=np.shape(x)
        mu=np.zeros(shape_x[0])
        for i in range(0,shape_x[0]):
             try:
                inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
             except:
                print("Singular matrix when computing pred. Small identity added.")
                inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
             finally:
                mu[i]=self.mu0+((np.array(self.k_t(x[i])).T).dot(inv)).dot(self.y[0:self.t])
                return mu

    def run_onestep(self,iter):
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
                    if(sigma_square<0):
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
                    if(sigma_square<0):
                        print("LCB error: sigma2=",end="")
                        print(sigma_square)
                        print("Let sigma2=0")
                        sigma_square=0
                return mu-np.sqrt(self.beta[self.t]*sigma_square)
            x=np.zeros(self.D)
            delta=np.zeros(self.D)
            for i in range(0,iter):
                x=ZOSGA(ucb,x,0.3,0.05,2)-delta
                delta=ZOSGD_bounded(ucb,x,self.bound,0.05,0.05,2)
            delta=ZOSGD_bounded(lcb,x,self.bound,0.05,0.05,50)
            self.x[self.t]=x+delta
            #while not (np.unique(self.x[range(0,self.t+1)])==self.x[range(0,self.t+1)]):
            #    self.x[self.t]=self.x[self.t]+np.random.multivariate_normal(np.zeros(self.D),0.01*np.identity(self.D))
            print("selected x+delta=",end="")
            print(x+delta)
            max_value=self.observe(self.x[self.t])
            print("function value=",end="")
            print(max_value)
            self.results[self.t-1]=max_value#note that t has increased
        else:
            print("t error!")
            return 0

    def run(self):
        for i in range(0,self.init_num):
            self.init()
        print("Init done")
        for i in range(0,self.T-self.init_num):
            print("##################################################################")
            print(i+1,end="")
            print("/",end="")
            print(self.T-self.init_num)
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
        print("Done.")
        print("##################################################################")
        index=np.where(self.results==np.max(self.results))
        print("Decision:",end="")
        print(self.x[index][0])
        print("Max value=",end="")
        print(self.results[index])
        print("Error:",end="")
        print(self.dis_fun(self.x[index][0],0.5*np.ones(self.D)))
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
    #print(ZOSGD_bounded(test,[30,30],[[-50,50],[-50,50]],1,0.01,200))
    #print(ZOSGD(test,[30,30],1,0.01,200))
    optimizer=STABLEOPT(beta=4*np.ones(30),init_num=20,mu0=0,epsilon=0.2,D=3,iter=50)
    optimizer.run()
    optimizer.print()