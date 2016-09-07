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
import os
import csv
from itertools import chain


class DataGenerator(object):
    
    
    def __init__(self,st=3,qu=5,ti=4,kc=2,direc=os.getcwd()):
        stu_id = 0;
        que_id = 0;
        self.N = st
        self.Q = qu
        self.T = ti
        self.K = kc
        self.path = direc
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
        data = [[[-1 for j in range(self.Q)] for j in range(self.N)] for t in range(self.T)]
        z = [[[0 for i in range(self.Q)] for j in range(self.N)] for t in range(self.T)]
        x =  [[[0 for k in range(self.K)] for j in range(self.N)] for t in range(self.T)]
        theta = [0 for j in range(self.N)]
        Diff = [0 for i in range(self.Q)]
        for t in range(0,self.T):
            for j in range(self.N):
                theta[j] = self.students[j].smartness()
                #index = np.random.randint(self.Q)
                for index in range(self.Q):
                    Diff[index] = self.questions[index].diff
                    sumK = 0
                    for k in range(self.K):
                        if(t == 0):
                            x[t][j][k] = np.random.uniform(0,1,1)
                        else:
                            noise =np.random.normal(0, 0.01,1)
                            x[t][j][k] = self.students[j].learningCurve()[k]* x[t-1][j][k]+self.questions[index].questionLearning()[k][t]+noise
                        sumK = self.questions[index].questionKnowledge()[k]*x[t][j][k]
                    noise1 =np.random.normal(0, 0.01,1)
                    z[t][j][index] = sumK+self.students[j].smartness()-self.questions[index].diff+noise1
                    p = 1/(1+math.exp(-z[t][j][index]))
                    rand  = np.random.uniform(0,1,1)
                    if (rand < p):
                        data[t][j][index]  = 1
                    else:
                        data[t][j][index]  = 0
        x = list(chain.from_iterable(x))
        dataTemp = np.array(x)
        fileName =dataTemp.transpose(0,1,2).reshape(self.T,-1)
        f = open(self.path+'KalmanFilter_X.csv', 'wb')
        w = csv.writer(f)
        w.writerows(fileName)
        f.close()
        f1 = open(self.path+'Smartnesss.csv', 'wb')
        w1 = csv.writer(f1)
        w1.writerows(theta)
        f1.close()
        f2 = open(self.path+'Difficulties.csv', 'wb')
        w2 = csv.writer(f2)
        w2.writerows(Diff)
        f2.close()
        return data
            
    def FactorialHMM(self):
        pf = 0.1
        data = [[[-1 for j in range(self.Q)] for j in range(self.N)] for t in range(self.T)]
        z = [[[0 for i in range(self.Q)] for j in range(self.N)] for t in range(self.T)]
        x =  [[[0 for k in range(self.K)] for j in range(self.N)] for t in range(self.T)]
        for t in range(0,self.T):
            for j in range(self.N):
                #index = np.random.randint(self.Q)
                for index in range(self.Q):
                    sumK = 0
                    for k in range(self.K):
                        if(t == 0):
                            x[t][j][k] =np.random.randint(2)
                        else:
                            rho = min(1,self.students[j].learningCurve()[k]+self.questions[index].questionLearning()[k][t])
                            rand = np.random.uniform(0,1,1)
                            if x[t-1][j][k] == 0:
                                if (rand <rho):
                                    x[t][j][k] = 1
                                else:
                                    x[t][j][k] = 0
                            elif x[t-1][j][k] == 1:
                                if (rand < pf):
                                    x[t][j][k] = 0
                                elif (rand < rho):
                                   x[t][j][k] = 2
                                else:
                                    x[t][j][k] = 1
                            elif x[t-1][j][k] == 2:
                                if (rand < pf):
                                    x[t][j][k] = 1
                                elif (rand < rho):
                                   x[t][j][k]= 3
                                else:
                                   x[t][j][k] = 2
                            else:
                                if (rand <pf):
                                    x[t][j][k] = 2
                                else:
                                    x[t][j][k] = 3
                        sumK = self.questions[index].questionKnowledge()[k]*x[t][j][k]
                    z[t][j][index] = sumK+self.students[j].smartness()-self.questions[index].diff
                    p = 1/(1+math.exp(-z[t][j][index]))
                    rand  = np.random.uniform(0,1,1)
                    if (rand < p):
                        data[t][j][index]  = 1
                    else:
                        data[t][j][index]  = 0
        dataTemp = np.array(x)
        fileName = dataTemp.transpose(0,1,2).reshape(self.T,-1)
        f = open(self.path+'FactorialHHM_X.csv', 'wb')
        w = csv.writer(f)
        w.writerows(fileName)
        f.close()
        return data         

class Student(DataGenerator):
    
    def __init__(self,s_id,kc,mu=1.0,sig = 0.2):
        self.id = s_id
        self.theta = np.random.normal(mu, sig,1)
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
    direc = "D:/Dissertation/Codes/Python/SynthData/"
    N=3 #number of students
    Q=5 #number of questions
    T=4 #number of time slots
    K=2 #number of knowledge components
    kf = DataGenerator(N,Q,T,K,direc)
    data1 = kf.kalmanFilter()
    data = np.array(data1)
    fileName = data.transpose(0,1,2).reshape(T,-1)
    f = open(direc+'KalmanFilter.csv', 'wb')
    w = csv.writer(f)
    w.writerows(fileName)
    f.close()
    print("Done! Writing Sysnthetic Data-Kalman Filter to the file ...")
    data2 = kf.FactorialHMM()
    data = np.array(data2)
    fileName = data.transpose(0,1,2).reshape(T,-1)
    f = open(direc+'FactorialHiddenMarkovModel.csv', 'wb')
    w = csv.writer(f)
    w.writerows(fileName)
    f.close()
    print("Done! Writing Sysnthetic Data-Factorial Hidden Markov Model to the file ...")

    
if __name__=="__main__":
    main()
