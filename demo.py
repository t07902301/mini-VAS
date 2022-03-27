# def get_rsp(point):
#     pass
import imp
import numpy as np
import pandas as pd
from random import randint, random,seed
from utils import *
from time import time
seed(0)
epsilon=3.2147064340016937e-07
# epsilon=4.597360999967811e-09

def proximity(point_1,point_2):
    try:
        # if np.sum(np.where((point_1==point_2),0,1))==0:
        #     return 0
        return np.exp(-np.power(np.linalg.norm(point_1-point_2),2)/epsilon)
    except:
        print(point_1,point_2)
        exit(0)
def expand(r,test_point):
    rsp=0
    for index,each in enumerate(r):
        rplc_point,rplc_rsp=each
        tmp_rsp=proximity(test_point,rplc_point)
        rplc_rsp+=tmp_rsp
        rsp+=tmp_rsp
        r[index]=(rplc_point,rplc_rsp)
    r.append((test_point,rsp))
    return r
def max_rsp(r):
    rsp_max=0
    rsp_max_index=0
    for index,each in enumerate(r):
        point,rsp=each
        if rsp_max<rsp:
            rsp_max=rsp
            rsp_max_index=index
    return rsp_max_index
def shrink(r):
    rsp_max_index=max_rsp(r)
    removed_point=r[rsp_max_index][0]
    del r[rsp_max_index]
    for index,each in enumerate(r):
        point,rsp=each
        rsp-=proximity(removed_point,point)
        r[index]=(point,rsp)
    return r
def interchange(point_set,k):
    r=[]
    for point in point_set:
        if len(r)<k:
            r=expand(r,point)
        else:
            r=expand(r,point)
            r=shrink(r)
    s=[point[0] for point in r]
    return np.array(s)
    # print(s)
def ReservoirSample(point_set,sample_size):
    pool=point_set[:sample_size]
    for i,point in enumerate(point_set[sample_size:]):
        j=randint(0,i)
        if j<sample_size:
            pool[j]=point
    # print(pool)
    return np.array(pool)
def Stratified_sampling(point_set,sample_size,bin_num=100):
    # point_set_size=point_set.shape[0]
    bin_size=point_set.shape[0]//bin_num
    sample_num_per_bin=sample_size//bin_num
    bin_samples=[]
    for i in range(bin_num):
        bin=point_set[i*bin_size:(i+1)*bin_size]
        bin_samples.append(ReservoirSample(bin,sample_num_per_bin))
    samples=bin_samples[0]
    for i in range(1,bin_num):
        # print(bin_samples[i].shape)
        samples=np.concatenate((samples,bin_samples[i]))
    # print(samples)
    return samples
        
if __name__=='__main__':
    # pass
    df=pd.read_csv('/users/yiwei/data/Data/000/Trajectory/20081023025304.plt',sep=',',names=['Latitude','Longitude','0','1','2','3','4'],skiprows=6)
    # tmp=df.loc[:3,['Longitude','Latitude']].values.tolist()
    # print(tmp)
    point_set=np.array(df.loc[:900,['Longitude','Latitude']].values.tolist())
    # print(point_set[:3])
    # epsilon= np.power(get_epsilon(point_set),2)*2
    # print(epsilon)
    start=time()
    a=ReservoirSample(point_set,100)
    # print(a)
    end=time()
    print(end-start)
    # ReservoirSample(point_set,2)
    # a=Stratified_sampling(point_set,100,10)
    # print(a.shape)
