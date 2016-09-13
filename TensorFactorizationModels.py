# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 23:09:38 2016
Tensor Factorizatoon Models
@author: afarasat
"""

from __future__ import division
import pandas as pd
import numpy as np
from decimal import *
import math
import os
import csv
from itertools import chain
import syntheticDataGenerator as sdg
import sys
import matplotlib.pyplot as plt

class TensorFactorizationModel(object):
    
    def __init__(self,path,fName,spRate = 0.40,syn = True):
        self.dir = path
        self.fileName = fName
        self.sparsity = spRate
        self.synData = syn
        
        
    def readFromFile(self):
        if (self.synData):
            self.data = np.genfromtxt(self.dir+self.fileName,delimiter=',')
            self.originalData = np.genfromtxt(self.dir+self.fileName,delimiter=',')
            dim = np.shape(self.data)[1]#*np.shape(self.data)[2]
            for t in range(len(self.data)):
                for i in range(dim):
                    rv = np.random.uniform(0,1,1)
                    if( rv < self.sparsity):
                        self.data[t,i] = -1
            return self.data
        else:
            return self.data
    
    def initialization(self, T,N,Q,K,ItMax):
        np.random.seed(19227347)
        self.W = 10*np.random.rand(K,Q)
#        for k in range(K):
#            for i in range(Q):
#                ranNu = np.random.uniform(0,1,1)
#                if ranNu < 0.2:
#                    self.W[k,i] = 0
        self.C =   6*np.random.rand(K,N)-3
        self.V =   4*np.random.rand(K, T)-2
        #print(self.W)
        self.CC =   np.random.rand(T, K, N)
        self.Theta =  np.zeros(shape = (N))
        self.Diff =  np.zeros(shape = (Q))
        self.lambda0 = 0.5;
        self.lambda1 = .5;
        self.lambda2 =0.2;
        self.NumIt = ItMax
        self.beta = 0.1
        self.gamm = 0.3
        self.K = K
    
    def ADMM(self,T,N,Q,K,ItMax):
        epoc = 0
        y = self.data
        objFunc = [0 for x in range(ItMax)]
        objFuncPlot = np.empty(shape=[ItMax+1])
        objFuncNow = self.ObjFunction(T,Q,N,K)
        objFuncPlot[epoc] = objFuncNow
        bestObj = sys.float_info.max
        bestIter = ItMax
        zero = np.zeros(shape=[K,Q])
        deltaW = np.empty(shape=[K, Q])
        deltaC = np.empty(shape=[K, N])
        deltaV = np.empty(shape=[K, T])
        while (epoc < ItMax):
            for i in range(Q):
                deltaW[:,i]=self.calculateDeltaW(i,T,Q,N,K)
                self.Diff[i] = self.Diff[i] - self.beta*self.calculateDiff(i,T,Q,N,K)
            self.W = self.W-self.beta*deltaW
            self.W =np.maximum(zero,self.W)
            for j in range(N):
                deltaC[:,j] = self.calculateDeltaTheta(j,T,Q,N,K)
                self.Theta[j] = self.Theta[j]-self.beta*self.calculateDeltaTheta(j,T,Q,N,K)
            self.C = self.C- self.beta*deltaC
            self.C =  (1/(1+self.gamm*self.beta))*self.C
            for t in range(T):
                deltaV[:,t] = self.calculateDeltaV(t,T,Q,N,K)
            objFuncNow = self.ObjFunction(T,Q,N,K)
            if ~math.isnan(objFuncNow) and objFuncNow < bestObj :
                bestObj = objFuncNow
                bestIter = epoc
                objFunc[epoc] = bestObj
                predictedData = self.predict(T,Q,N,K)
            else:
                objFunc[epoc] = objFunc[epoc-1]

            #self.beta = 2/(epoc+2)
            #print(objFuncNow)
            epoc = epoc +1
            objFuncPlot[epoc] = objFuncNow
        print(sum(sum(np.abs(predictedData-self.originalData)))/(T*N*Q))
        print(bestIter)
        print(self.W)
        print(self.C)
        print(self.V)
        x = range(ItMax+1)
        plt.plot(x,objFuncPlot,color='black',label='ADMM')
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        #print(np.linalg.norm(self.originalData-predictedData,'fro'))
        
    def BCD(self,T,N,Q,K,ItMax):
        epoc = 0
        y = self.data
        objFunc = [0 for x in range(ItMax)]
        objFuncNow = self.ObjFunction(T,Q,N,K)
        bestObj = sys.float_info.max
        objFuncPlot = np.empty(shape=[ItMax+1])
        objFuncPlot[epoc] = objFuncNow
        bestIter = ItMax
        while (epoc < ItMax):
            i = np.random.randint(0,Q,1)
            j = np.random.randint(0,N,1)
            t = np.random.randint(0,T,1)
            deltaWi_ = self.calculateDeltaW(i,T,Q,N,K)
            deltaCj_ = self.calculateDeltaC(j,T,Q,N,K)
            deltaVt_ = self.calculateDeltaV(t,T,Q,N,K)
            deltaTheta_j = self.calculateDeltaTheta(j,T,Q,N,K)
            deltaDiff_i = self.calculateDiff(i,T,Q,N,K)
            for k in range(K):
                self.W[k,i] = self.W[k,i] - self.beta*deltaWi_[k]
                self.W[k,i] = max(0,self.W[k,i]-self.lambda1*self.beta)
                #self.W[k,i] = min(5,self.W[k,i])
                #self.C[k,j] = max(self.C[k,j] - self.beta*deltaCj_[k],3)
                self.C[k,j] = (1/(1+self.gamm*self.beta))*self.C[k,j]
                self.V[k,t] = self.V[k,t] - self.beta*deltaVt_[k]
                #self.V[k,j] = (1/(1+self.gamm*self.beta))*self.V[k,j]
            self.Theta[j] =self.Theta[j]-self.beta*deltaTheta_j
            #self.Theta[j] = max(self.Theta[j]-self.beta*deltaTheta_j,0)
            self.Diff[i] = self.Diff[i] - self.beta*deltaDiff_i
            objFuncNow = self.ObjFunction(T,Q,N,K)
            if ~math.isnan(objFuncNow) and objFuncNow < bestObj :
                bestObj = objFuncNow
                bestIter = epoc
                objFunc[epoc] = bestObj
                predictedData = self.predict(T,Q,N,K)
            else:
                objFunc[epoc] = objFunc[epoc-1]

            self.beta = 2/(epoc+2)
            #print(objFuncNow)
            epoc = epoc +1
            objFuncPlot[epoc] = objFuncNow            
            
        print(sum(sum(np.abs(predictedData-self.originalData)))/(T*N*Q))
        print(bestIter)
        print(self.W)
        print(self.C)
        print(self.V)
        x = range(ItMax+1)
        plt.plot(x,objFuncPlot,label='GD')
        #print(np.linalg.norm(self.originalData-predictedData,'fro'))
        
        

    def predict(self,T,Q,N,K):
        y_hat = np.zeros((T,N*Q))
        for t in range(T):
            for j in range(N):
                for i in range(Q):
                    phi = 1/(1+self.exponentialTerm(i,j,t))
                    #rand  = np.random.uniform(0,1,1)
                    if (0.5 < phi):
                        y_hat[t,j*(Q-1)+i] = 1
        return y_hat
                    
    def ObjFunction(self,T,Q,N,K):
        sumTotal = 0
        for t in range(T):
            for i in range(Q):
                for j in range(N):
                    yijt = self.data[t,j*(Q-1)+i]
                    if(yijt != -1.0):
                        sumTotal = sumTotal + yijt*np.log(1/(1+self.exponentialTerm(i,j,t))+1)+(1-yijt)*(1-np.log(1/(1+self.exponentialTerm(i,j,t))+1.0))
        sumTotal = -sumTotal +  self.lambda0*np.linalg.norm(self.W,1)+self.lambda2*np.linalg.norm(self.C,ord='fro')+ self.lambda1*np.linalg.norm(self.W)
       
        return sumTotal


        
    def calculateDeltaW(self,i,T,Q,N,K):
        delta = np.zeros(shape=(K))
        for k in range(K):
            sumK = 0
            for j in range(N):
                for t in range(T):
                    yijt = self.data[t,j*(Q-1)+i]
                    if(yijt != -1.0):
                        #yijt = np.random.randint(0,2,1)
                        sumK = sumK + self.C[k,j]*self.V[k,t]*(yijt-1/(1+self.exponentialTerm(i,j,t)))
            delta[k] = -sumK+(self.lambda1*self.W[k,i])/(np.linalg.norm(self.W)**0.5)
        return delta
            
    def calculateDeltaC(self,j,T,Q,N,K):
        #print(Q)
        delta = np.zeros(shape=(K))
        for k in range(K):
            sumK = 0
            for i in range(Q):
                for t in range(T):
                    yijt = self.data[t,j*(Q-1)+i]
                    if(yijt != -1.0):
                        #yijt = np.random.randint(0,2,1)
                        sumK = sumK + self.W[k,i]*self.V[k,t]*(yijt-1/(1+self.exponentialTerm(i,j,t)))
            delta[k] = -sumK+(self.lambda2*self.C[k,j])/(np.linalg.norm(self.C)**0.5)
        return delta       
        
    def calculateDeltaV(self,t,T,Q,N,K):
        #print(K)
        delta = np.zeros(shape=(K))
        for k in range(K):
            sumK = 0
            for i in range(Q):
                for j in range(N):
                    yijt = self.data[t,j*(Q-1)+i]
                    if(yijt != -1.0):
                        #yijt = np.random.randint(0,2,1)
                        sumK = sumK + self.W[k,i]*self.C[k,j]*(yijt-1/(1+self.exponentialTerm(i,j,t)))
            delta[k] = -sumK
        return delta
        
    def calculateDeltaTheta(self,j,T,Q,N,K):
        sumT = 0
        for i in range(Q):
            for t in range(T):
                yijt = self.data[t,j*(Q-1)+i]
                if(yijt != -1.0):
                    #yijt = np.random.randint(0,2,1)
                    sumT = sumT+(yijt-1/(1+self.exponentialTerm(i,j,t)))
        return -sumT
    
    def calculateDiff(self,i,T,Q,N,K):
        sumD = 0
        for j in range(N):
            for t in range(T):
                yijt = self.data[t,j*(Q-1)+i]
                if(yijt != -1.0):
                    #yijt = np.random.randint(0,2,1)
                    sumD = sumD+(yijt-1/(1+self.exponentialTerm(i,j,t)))
        return sumD
        
    def exponentialTerm(self,i,j,t):
        sumK = 0
        for k in range(self.K):
            sumK = sumK+self.W[k,i]*self.C[k,j]*self.V[k,t]
        return np.exp(-sumK-self.Theta[j]+self.Diff[i])
def main():
    path =  "D:/Dissertation/Codes/Python/SynthData/"
    fileName1 = 'KalmanFilter.csv'
    fileName2 = 'FactorialHiddenMarkovModel.csv'
    ItMax = 500
    N=3 #number of students
    Q=5 #number of questions
    T=4 #number of time slots
    K=2 #number of knowledge components
    kf = sdg.DataGenerator(N,Q,T,K,path)
    data1 = kf.kalmanFilter()
    data = np.array(data1)
    fileN = data.transpose(0,1,2).reshape(T,-1)
    f = open(path+'KalmanFilter.csv', 'wb')
    w = csv.writer(f)
    w.writerows(fileN)
    f.close()
    print("Done! Writing Sysnthetic Data-Kalman Filter to the file ...")
    data2 = kf.FactorialHMM()
    data = np.array(data2)
    fileName = data.transpose(0,1,2).reshape(T,-1)
    f = open(path+fileName2, 'wb')
    w = csv.writer(f)
    w.writerows(fileName)
    f.close()
    print("Done! Writing Sysnthetic Data-Factorial Hidden Markov Model to the file ...")
    spRate =0.6
    tenModel = TensorFactorizationModel(path,fileName1,spRate,True) 
    tenModel.readFromFile()
    tenModel.initialization(T,N,Q,K,ItMax)
    tenModel.BCD(T,N,Q,K,ItMax)
    tenModel.initialization(T,N,Q,K,ItMax)
    tenModel.ADMM(T,N,Q,K,ItMax)

    
    
    
if __name__=="__main__":
    main()

