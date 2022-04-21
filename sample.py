'''
A package for Interchange, Resevior and Stratified
'''

import numpy as np
import pandas as pd
# from cvxopt import glpk,matrix
from random import randint, random,seed
from utils import *
import time

class interchange:
    def __init__(self,prox:proximity) -> None:
        self.proximity=prox
    
    def expand(self,r,test_point):
        rsp=0
        for index,each in enumerate(r):
            rplc_point,rplc_rsp=each
            tmp_rsp=self.proximity.run(test_point,rplc_point)
            rplc_rsp+=tmp_rsp
            rsp+=tmp_rsp
            r[index]=(rplc_point,rplc_rsp)
        r.append((test_point,rsp))
        return r
    def max_rsp(self,r):
        rsp_max=0
        rsp_max_index=0
        for index,each in enumerate(r):
            point,rsp=each
            if rsp_max<rsp:
                rsp_max=rsp
                rsp_max_index=index
        return rsp_max_index
    def shrink(self,r):
        rsp_max_index=self.max_rsp(r)
        removed_point=r[rsp_max_index][0]
        del r[rsp_max_index]
        for index,each in enumerate(r):
            point,rsp=each
            rsp-=self.proximity.run(removed_point,point)
            r[index]=(point,rsp)
        return r
    def run(self,point_set,k):
        r=[]
        for point in point_set:
            if len(r)<k:
                r=self.expand(r,point)
            else:
                r=self.expand(r,point)
                r=self.shrink(r)
        s=[point[0] for point in r] #(coordinates,responsibility) for each point in r
        return np.array(s)
def ReservoirSample(point_set,sample_size):
    seed(0)
    pool=point_set[:sample_size]
    for i,point in enumerate(point_set[sample_size:]):
        j=randint(0,i)
        if j<sample_size:
            pool[j]=point
    return np.array(pool)
def Stratified_sampling(point_set,sample_size,bin_num=100):
    seed(0)
    # point_set_size=point_set.shape[0]
    bin_size=point_set.shape[0]//bin_num
    sample_size_per_bin=sample_size//bin_num
    bin_samples=[]
    for i in range(bin_num):
        bin=point_set[i*bin_size:(i+1)*bin_size]
        bin_samples.append(ReservoirSample(bin,sample_size_per_bin))
        
    samples=bin_samples[0]
    for i in range(1,bin_num):
        # print(bin_samples[i].shape)
        samples=np.concatenate((samples,bin_samples[i]))
    # print(samples)
    return samples
class interchange_timer:
    def __init__(self,prox:proximity) -> None:
        self.proximity=prox
    
    def expand(self,r,test_point):
        rsp=0
        for index,each in enumerate(r):
            rplc_point,rplc_rsp=each
            tmp_rsp=self.proximity.run(test_point,rplc_point)
            rplc_rsp+=tmp_rsp
            rsp+=tmp_rsp
            r[index]=(rplc_point,rplc_rsp)
        r.append((test_point,rsp))
        return r
    def max_rsp(self,r):
        rsp_max=0
        rsp_max_index=0
        for index,each in enumerate(r):
            point,rsp=each
            if rsp_max<rsp:
                rsp_max=rsp
                rsp_max_index=index
        return rsp_max_index
    def shrink(self,r):
        rsp_max_index=self.max_rsp(r)
        removed_point=r[rsp_max_index][0]
        del r[rsp_max_index]
        for index,each in enumerate(r):
            point,rsp=each
            rsp-=self.proximity.run(removed_point,point)
            r[index]=(point,rsp)
        return r
    def run(self,point_set,k,timeout,stop_points):
        # global pool
        pool=[]
        i=0
        idx=0
        set_size=point_set.shape[0]
        print(set_size)
        while i<stop_points:
            i+=1
            start=time.time()
            # for point in point_set:
            while idx<set_size:
                if time.time()-start>timeout:
                    print(time.ctime())
                    break
                point=point_set[idx]
                if len(pool)<k:
                    pool=self.expand(pool,point)
                else:
                    pool=self.expand(pool,point)
                    pool=self.shrink(pool)
                idx+=1
            print(idx,len(pool))
            s=[point[0] for point in pool] #(coordinates,responsibility) for each point in r
            np.save('stop_points/int_{}_{}_{}_{}.npy'.format(set_size,timeout,i,k),s)
            print('save {}th file'.format(i))
        return np.array(s)
        
if __name__=='__main__':
    df=pd.read_csv('data/Data/000/Trajectory/20081023025304.plt',sep=',',names=['Latitude','Longitude','0','1','2','3','4'],skiprows=6)
    # tmp=df.loc[:3,['Longitude','Latitude']].values.tolist()
    # print(tmp)
    point_set=np.array(df.loc[:6,['Longitude','Latitude']].values.tolist())
    # print(point_set[:3])
    # epsilon= np.power(get_epsilon(point_set),2)*2
    # print(epsilon)
    prox=proximity(point_set,set_eps=False)
    start=time()
    int_sample=interchange(prox)
    a=int_sample.run(point_set,2)
    # ilp_sample=ilp(prox)
    # a=ilp_sample.run(point_set,[10])
    end=time()
    print('time: {} min {} s'.format((end-start)//60,(end-start)%60))
    # print(a)
    # print(prox.epsilon)

    # ReservoirSample(point_set,2)
    # a=Stratified_sampling(point_set,100,10)
    # print(a.shape)

class ilp: 
    '''
    Deprecated ILP wrapper
    '''
    def __init__(self,prox:proximity) -> None:
        self.proximity=prox
    def run(self,points,sample_size):
        # sample_size=100
        num_a=len(points)
        num_b=num_a*(num_a-1)//2
        # m=2*num_a+num_b*5+2
        # m=2*num_a+num_b*5
        m=num_b*4+num_a
        # n=num_b+num_a+1
        n=num_b+num_a
        # G=[[0 for col in range(n)] for row in range(m)]
        g_row=np.repeat([0],n)
        G=np.repeat([g_row],m,axis=0)    
        A=[0 for row in range(n)]
        h=[0 for i in range(m)]

        a_combinations=[]
        for i in range(num_a-1):
            for j in range(i+1,num_a):
                a_combinations.append((i,j))
        c_b=[self.proximity.run(points[a_combinations[i][0]],points[a_combinations[i][1]]) for i in range(num_b)]
        c_a=[0 for i in range(num_a)]
        # c_k=[0]
        # c=c_b+c_a+c_k
        c=c_b+c_a
        for b_index in range(num_b):
            G[b_index][b_index]=1
            G[b_index][a_combinations[b_index][0]+num_b]=-1
        for b_index in range(num_b):
            row=num_b+b_index
            G[row][b_index]=1
            G[row][a_combinations[b_index][1]+num_b]=-1
        for b_index in range(num_b):
            row=b_index+2*num_b
            G[row][b_index]=-1
            G[row][a_combinations[b_index][0]+num_b]=1
            G[row][a_combinations[b_index][1]+num_b]=1
            h[row]=1
        for b_index in range(num_b):
            row=b_index+3*num_b
            G[row][b_index]=-1
        for a_index in range(num_a):
            row=a_index+4*num_b
            G[row][a_index+num_b]=-1

        for i in range(num_b,num_b+num_a):
            A[i]=1
    
        # b=[sample_size]

        G=matrix(np.array(G,dtype=float))
        c=matrix(np.array(c,dtype=float))
        h=matrix(np.array(h,dtype=float))
        A=matrix(np.array(A,dtype=float),(1,n))
        samples_list=[]
        for size in sample_size:
            b=[size]
            b=matrix(np.array(b,dtype=float))
            # print(G.size,c.size,h.size,A.size,b.size)
            # print(num_a,num_b)
            # print(G)
            # print(h)
            (status, sol) =glpk.ilp(c=c,   # c parameter
                                    G=G,     # G parameter
                                    h=h,     # h parameter
                                    A=A,
                                    b=b,
                                    I=set(range(0, len(c))),
                                    B=set(range(0, len(c)))
                                    )
            print(status)
            sample_b_idx=[]
            for index,each in enumerate(sol[:num_b]):
                if each ==1:
                    # print(index)
                    sample_b_idx.append(index)
            samples=[]
            for idx in sample_b_idx:
                samples.append(a_combinations[idx][0])
                samples.append(a_combinations[idx][1])
            samples_list.append(points[list(set(samples))])
        # print(G.size)
        # return points[list(set(samples))]
        return samples_list
