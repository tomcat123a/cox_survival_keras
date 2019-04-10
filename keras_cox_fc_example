#note that tf.print does not work in IPython condole
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input
from keras.models import Sequential
import numpy as np
import random
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Dropout,Activation,Flatten,Lambda
from keras.layers.normalization import BatchNormalization
import keras
import time
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from ann_visualizer.visualize import ann_viz
import glmnet_python
from glmnet import glmnet
import pandas as pd
import scipy
import os
import sys
from numpy.random import seed
from tensorflow import set_random_seed
from keras import regularizers
from keras.engine.topology import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec

os.environ["PATH"] += os.pathsep + 'C:\Program Files\R\R-3.4.2\\bin'
os.chdir('C:\python_study\surviv')





def gen_x(n,p,rho):
    if abs(rho) < 1 :
        beta=np.sqrt(rho/(1-rho))
        x0=np.random.normal(size=(n,p))
        z=np.random.normal(size=(n,1))
        x=beta*np.repeat(z,repeats=p,axis=1)+x0
  
    if abs(rho)==1:
        x=np.repeat(z,repeats=p,axis=1)
    return x

 
## This function creates true survival times as described in section 3 of the paper. In all simulations we set snr (signal to noise ratio) to 3.
np.random.seed(0)
genecoef_fixed = np.random.rand(6)
def genecoef(p,rp=6):
    #return list( map(lambda x : np.power(-1,x)*np.exp(-0.1*(x-1)), np.arange(1,p+1,1)) )
    return list( genecoef_fixed ) + list( np.zeros(p-rp))

def gen_times(x,snr):
    n,p=x.shape
    coef=genecoef(p)
    f=np.matmul(np.matrix(x),np.matrix(coef).T)
    e=np.random.normal(size=(n,1))
    k=np.sqrt(np.var(f)/(snr*np.var(e)))
    y=np.exp(f+k*e)
    return(y)


## This function creates true survival times as described in section 3 of the paper. In all simulations we set snr (signal to noise ratio) to 3.

def gen_times_censor(x,snr):
    n,p=x.shape
    coef=genecoef(p)
    f=np.matmul(np.matrix(x),np.matrix(coef).T)
    e=np.random.normal(size=(n,1))
    k=np.sqrt(np.var(f)/(snr*np.var(e)))
    y=np.exp(k*e)
    return(y)


def nltr2(x):
    y1 = x[:,0]*x[:,1]
    y2 = x[:,2]*x[:,3]
    y3 = x[:,4]**2
    y4 = 0.5 * np.exp(x[:,5] + 2 * x[:,6])
    y5 = log(0.5 + abs(x[:,7]))
    y6 = 0.1*np.exp(x[:,8]* x[:,9])
    newx = np.column_stack((y1,y2,y3,y4,y5,y6))
    return newx
def nltr(x):
    y1 = x[:,0]*x[:,1]
    y2 = x[:,2]*x[:,3]
    y3 = x[:,4]**2
    y4 = x[:,5]* (x[:,6]**2)
    y5 = x[:,7]*x[:,8]* x[:,9]
    y6 = 0.5 *np.exp(x[:,8]* x[:,9])
    newx = np.column_stack((y1,y2,y3,y4,y5,y6,x[:,10:]))
    return newx
    

def survdata(n,p,snr,rho):
    x = gen_x(n,p,rho)
    time = gen_times(x,snr)
    censortime = gen_times_censor(x,snr)
    y = np.apply_along_axis(np.min,1,np.column_stack((time,censortime)))
    y = np.array(y)
    #b==0 censored b ==1 uncensored
    b = np.apply_along_axis(np.argmax,1,np.column_stack((time,censortime)))
    b = np.array(b)
    a = x
    ordery=np.argsort(y)
    a=a[ordery]
    y=y[ordery]
    b=b[ordery]
    Rlist=[]
    event_index=np.argwhere(b==1).ravel().astype(np.int32)
    nsample=len(b)
    nevent=sum(b)
    Rlist=[]
    for j in range(nevent):
        Rlist+=[ list(range(np.argwhere(b==1).ravel()[j],nsample) )]
    bmask = b.astype(bool)
    cumlist=list(reversed(np.append(event_index,n)))
    slarr=np.vectorize(lambda x:(len(x)-1))
    nctrue = np.sum(slarr(Rlist))
    #a:n(#samples)*p(#features) matrix,survival time from short to high
    #y:survival time
    #b censored(0) or not(1)
    #bmask bool(b)
    #nevent #uncensored
    return a,y,b,bmask,nsample,nevent,event_index,Rlist,cumlist,nctrue









def ploss(y_true,y_pred):
    z = 0
#    for j in event_index:
#        z = z + K.sum(y_pred[j,0])
#        z = z + K.constant(y_pred[j,0])
    #z = K.sum(tf.boolean_mask(y_pred,bmask) )   
    #iz = K.print_tensor(tf.boolean_mask(y_pred,bmask),'y_pred_mask is')
#    gz = K.print_tensor(K.gather(y_pred,event_index),'y_pred_gather is')
#    z = K.sum(gz)
    for j in Rlist:
        tempz=0
        for i in j:
            tempz = tempz + K.exp(y_pred[i,0])
        z = z - K.log(tempz)
    z = -z    
    return z




# =============================================================================
# def ploss(y_true,y_pred):
#     #y_pred for sample x_i is the value of np.dot(x_i,beta) in the linear cox case
#     #y_pred is the loss for sample i
#     z = 0
#     #for j in event_index:
#         #z = z + K.sum(y_pred[j,0])
#         #z = z + K.constant(y_pred[j,0])
#     #z = K.sum(tf.boolean_mask(y_pred,bmask) )   
#     #iz = K.print_tensor(tf.boolean_mask(y_pred,bmask),'y_pred_mask is')
#     #gz = K.print_tensor(K.gather(y_pred,event_index),'y_pred_gather is')
#     z = K.sum(K.gather(y_pred,event_index))
#     currentsum = 0
#     for j in range(nevent):
#         currentsum = currentsum + K.sum(K.exp(K.gather(y_pred,\
#         int(np.array(range(cumlist[j+1],cumlist[j]))  ))))
#         z = z - K.log(currentsum)
#         #tempz=0
#         #for i in j:
#             #tempz = tempz + K.exp(y_pred[i,0])
#         #z = z - K.log(tempz)
#     z = -z  
#     
#     return z
# =============================================================================



def c_index(y_true, y_pred):
    #y_pred is the loss for sample i
    c_hat = 0
    if b[-1]==1:
        num=nevent-1
    else:
        num=nevent
    for i in range(num):
        c_hat = c_hat + K.sum(K.cast(y_pred[event_index[i]+1:,0]\
        <y_pred[event_index[i],0],tf.float32))
        #c_hat = c_hat + K.sum(K.cast(y_pred[event_index[i]+1:,0]\
                                             #<y_pred[event_index[i],0],float32))
    return c_hat/nctrue 
def py_c_index(y_pred):
    #y_pred is the loss for sample i
    c_hat = 0
    if b.iloc[-1]==1:
        num=nevent-1
    else:
        num=nevent
    for i in range(num):
        c_hat = c_hat + np.sum(y_pred[event_index[i]+1:]\
        <y_pred[event_index[i]])
        #c_hat = c_hat + K.sum(K.cast(y_pred[event_index[i]+1:,0]\
                                             #<y_pred[event_index[i],0],float32))
    return c_hat/nctrue 


def getmodel(mod=0):
    if mod == 0:#linear
        model=Sequential()
        model.add(Dense(1,activation='linear',kernel_initializer='normal',\
                    input_dim=a.shape[1]))  
        return model
    if mod == 1:#deep
        model=Sequential()
        model.add(Dense(20,activation='sigmoid',kernel_initializer='normal',\
                        input_dim=a.shape[1]))
        #model.add(BatchNormalization())
        model.add(Dense(20,activation='sigmoid',kernel_initializer='normal'\
                        ))    
        #model.add(BatchNormalization())
        model.add(Dense(10,activation='sigmoid',kernel_initializer='normal'\
                        ))  
        #model.add(BatchNormalization())
        #model.compile(loss=ploss,optimizer='newton-raphson')
        model.add(Dense(1,activation='linear',kernel_initializer='normal'\
                        ))    
        return model
    if mod == 2:#deep sigmoid with bn and dropout 
        model=Sequential()
        model.add(Dense(20,activation='sigmoid',kernel_initializer='normal',\
                        input_dim=a.shape[1],kernel_regularizer=regularizers.l1(l_l1)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(20,activation='sigmoid',kernel_initializer='normal'\
                        ,kernel_regularizer=regularizers.l1(l_l1)))    
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(10,activation='sigmoid',kernel_initializer='normal'\
                        ,kernel_regularizer=regularizers.l1(l_l1)))  
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='linear',kernel_initializer='normal'\
                        ))
        return model
    if mod == 3:##deep relu with bn and dropout 
        model=Sequential()
        model.add(Dense(20,activation='relu',kernel_initializer='normal',\
                        input_dim=a.shape[1],kernel_regularizer=regularizers.l1(l_l1)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(20,activation='relu',kernel_initializer='normal'\
                        ,kernel_regularizer=regularizers.l1(l_l1)))    
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(10,activation='relu',kernel_initializer='normal'\
                        ,kernel_regularizer=regularizers.l1(l_l1)))  
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='linear',kernel_initializer='normal'\
                        ))
        return model
        
        

os.chdir('C:\python_study\surviv')
err=[]
k_vb=1
k_ep=10
n=1000
p=40
snr=3
rho=0.0
l_l1=0.05
for innerseed in range(5):
        
    for modj in range(4):
        
        seed(innerseed)
        set_random_seed(innerseed+1)
        a,y,b,bmask,nsample,nevent,event_index,Rlist,cumlist,nctrue= survdata(n,p,snr,rho)
        
        sc=StandardScaler()
        a=nltr(a)
        a=sc.fit_transform(a)
        
        if modj==0:
            coxx = pd.DataFrame(a)
            coxy = pd.DataFrame( np.column_stack((y,b)) )
            coxy = coxy.rename(columns={0:'time',1:'status'})
            coxx.to_csv('C:\python_study\surviv\cx.csv',index=False)
            coxy.to_csv('C:\python_study\surviv\cy.csv',index=False)
            
        # model
        model = getmodel(modj)
        model.compile(loss=ploss1,optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
        epsilon=None, decay=0.0, amsgrad=False),metrics=[c_index])
        model.fit(x=a,y=y,batch_size=len(a),epochs=k_ep,verbose=k_vb,shuffle=False)
        testa,y,b,bmask,nsample,nevent,event_index,Rlist,cumlist,nctrue= survdata(n,p,snr,rho)
        testa=nltr(testa)
        testa=sc.fit_transform(testa)
        model.compile(loss=ploss2,optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, \
        epsilon=None, decay=0.0, amsgrad=False),metrics=[c_index])
        
        err=err+[model.evaluate(x=testa, y=y,batch_size=len(testa))[1]]
        #deep network no batch
    
    os.system('Rscript C:\python_study\surviv\surviv.R')
    glmnet_coef1 = pd.read_csv('C:\python_study\surviv\coef1.csv')
    glmnet_coef2 = pd.read_csv('C:\python_study\surviv\coef2.csv')
    err=err+[py_c_index(np.matmul(np.matrix(a),np.matrix(glmnet_coef1)))]
    err=err+[py_c_index(np.matmul(np.matrix(a),np.matrix(glmnet_coef2)))]
    


en=np.array(err)
en=en.reshape((10,6))

mean_output=np.apply_along_axis(np.mean,0,en)
std_output=np.apply_along_axis(np.std,0,en)
print(mean_output)
print(std_output)
pd.DataFrame([mean_output,std_output]).to_csv('simuresult'+str(n)+'_'+str(p)+'_'+str(snr)+'.csv')

