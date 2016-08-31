# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 14:59:40 2016

@author: afarasat
Synthetic Data Generation
"""
from __future__ import division
import pandas as pd
import numpy as np
from decimal import *
import math

class DataGenerator(object):
    
    
    def __init__(self,st=2,qu=3,ti=4,kc=2):
        stu_id = 0;
        que_id = 0;
        self.N = st
        self.Q = qu
        self.T = ti
        self.K = kc
        mu_st, sigma_st = 1.0, 0.2 # mean and standard deviation
        #self.theta = max(0,np.random.normal(mu_st, sigma_st, self.N))
        mu_q, sigma_q = 1.2,0.4
        #self.diff = max(0,np.random.normal(mu_q, sigma_q, self.Q))
        self.learCur=[[]]
        self.students = []
        self.questions = []
        for j in range(0,self.N):
            self.students.append(Student(stu_id,self.K,mu_st, sigma_st))
            stu_id = stu_id + 1
        for i in range(0,self.Q):
            self.questions.append(Question(que_id,self.K,mu_q, sigma_q,self.T))
            que_id = que_id + 1
    
    def kalmanFilter(self):
        data = [[[-1 for t in range(self.T)] for j in range(self.N)] for i in range(self.Q)]
        z = [[[0 for t in range(self.T)] for j in range(self.N)] for i in range(self.Q)]
        x =  [[[0 for t in range(self.T)] for j in range(self.N)] for k in range(self.K)]
        for t in range(0,self.T):
            for j in range(self.N):
                #index = np.random.randint(self.Q)
                for index in range(self.Q):
                    sumK = 0
                    for k in range(self.K):
                        if(t == 0):
                            x[k][j][t] = np.random.uniform(0,1,1)
                        else:
                            noise =np.random.normal(0, 0.1,1)
                            x[k][j][t] = self.students[j].learningCurve()[k]* x[k][j][t-1]+self.questions[index].questionLearning()[k][t]+noise
                        sumK = self.questions[index].questionKnowledge()[k]*x[k][j][t]
                    noise1 =np.random.normal(0, 0.1,1)
                    z[index][j][t] = sumK+self.students[j].smartness()-self.questions[index].diff+noise1
                    p = 1/(1+math.exp(-z[i][j][t]))
                    rand  = np.random.uniform(0,1,1)
                    if (rand < p):
                        data[index][j][t]  = 1
                    else:
                        data[index][j][t]  = 0
        return data
            
    def FactorialHMM(self):
        pf = 0.1
        data = [[[-1 for t in range(self.T)] for i in range(self.N)] for j in range(self.Q)]
        z = [[[0 for t in range(self.T)] for i in range(self.N)] for j in range(self.Q)]
        x =  [[[0 for t in range(self.T)] for i in range(self.N)] for j in range(self.K)]
        for t in range(0,self.T):
            for j in range(self.N):
                #index = np.random.randint(self.Q)
                for index in range(self.Q):
                    sumK = 0
                    for k in range(self.K):
                        if(t == 0):
                            x[k][j][t] =np.random.randint(2)
                        else:
                            rho = min(1,self.students[j].learningCurve()[k]+self.questions[index].questionLearning()[k][t])
                            rand = np.random.uniform(0,1,1)
                            if x[k][i][t] == 0:
                                if (rand <rho):
                                    x[k][j][t] = 1
                                else:
                                    x[k][j][t] = 0
                            elif x[k][j][t] == 1:
                                if (rand < pf):
                                    x[k][j][t] = 0
                                elif (rand < rho):
                                    x[k][j][t] = 2
                                else:
                                     x[k][j][t] = 1
                            elif x[k][j][t] == 2:
                                if (rand < pf):
                                    x[k][j][t] = 1
                                elif (rand < rho):
                                    x[k][j][t] = 3
                                else:
                                     x[k][j][t] = 2
                            else:
                                if (rand <pf):
                                    x[k][j][t] = 2
                                else:
                                    x[k][j][t] = 3
                        sumK = self.questions[index].questionKnowledge()[k]*x[k][j][t]
                    z[index][j][t] = sumK+self.students[j].smartness()-self.questions[index].diff
                    p = 1/(1+math.exp(-z[i][j][t]))
                    rand  = np.random.uniform(0,1,1)
                    if (rand < p):
                        data[index][j][t]  = 1
                    else:
                        data[index][j][t]  = 0
        return data         

class Student(DataGenerator):
    
    def __init__(self,s_id,kc,mu=1.0,sig = 0.2):
        self.id = s_id
        self.theta = np.random.normal(mu, sigma,1)
        self.learCur = []
        for k in range(0,kc):
             self.learCur.append(np.random.uniform(0,1,1))
        
    def smartness(self):
        return self.theta
    def learningCurve(self):
        return self.learCur
    
class Question(DataGenerator):
    
    def __init__(self,q_id,kc,mu,sig,T):
        self.id = q_id
        self.diff = np.random.normal(mu, sig,1) 
        rand = np.random.uniform(0,1,1)
        t1 = T/3
        t2 = (2*T)/3
        c = (t1+t2)/2
        sigma = (t1+t2)/5
        self.imp =  [[0 for x in range(T)] for y in range(kc)] 
        a = [[0 for x in range(T)] for y in range(kc)] 
        self.w = [0 for x in range(kc)]
        if(rand<=0.33):
            for k in range(0,kc):
                for t in range(0,T):
                        if(t<t1):
                            a[k][t] = 1
                        elif  (t>=t1 and t<=(t1+t2)/2):
                            a[k][t] = 1-2*((t-t1)/(t2-t1))**2
                        elif  (t>=(t1+t2)/2 and t<=t2):
                            a[k][t]= 2*((t-t2)/(t2-t1))**2
                        else:
                            a[k][t] = 0.05*np.random.uniform(0,1,1)
        elif (rand>0.33 and rand<=0.66):
            for k in range(0,kc):
                for t in range(0,T):
                    a[k][t] = math.exp(-((t-c)**2)/sigma)
        else:
            for k in range(0,kc):
                for t in range(0,T):
                        if(t<t1):
                           a[k][t]= 0.05*np.random.uniform(0,1,1)                            
                        elif  (t>=t1 and t<=(t1+t2)/2):
                            a[k][t] = 2*((t-t1)/(t2-t1))**2
                        elif  (t>=(t1+t2)/2 and t<=t2):
                            a[k][t] = 1-2*((t-t2)/(t2-t1))**2
                        else:
                            a[k][t] = 1
                            
        for k in range(0,kc):
            rand = np.random.uniform(0,1,1)
            if (rand<0.2):
                self.w[k] = 1
            else:
                self.w[k] = 0
        self.imp = a 
        
    def questionLearning(self):
          return self.imp
          
    def questionKnowledge(self):
          return self.w
    def difficulty(self):
        return self.diff
def main():
    N=3 #number of students
    Q=5 #number of questions
    T=4 #number of time slots
    K=2 #number of knowledge components
    kf = DataGenerator(N,Q,T,K)
    data1 = kf.kalmanFilter()
    print(data1)
    print("*************************")
    data2 = kf.FactorialHMM()
    print(data2)

    
if __name__=="__main__":
    main()