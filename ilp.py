'''
A package for BLP/ILP(Integer Linear Programming)
'''
import numpy as np
import pandas as pd
from cvxopt import glpk,matrix
from time import time
# from utils import *
epsilon=3.2147064340016937e-07
def proximity(point_1,point_2):
    # if np.sum(np.where((point_1==point_2),0,1))==0:
    #     return 0
    return np.exp(-np.power(np.linalg.norm(point_1-point_2),2)/epsilon)
    # return     np.linalg.norm(point_1-point_2)/epsilon


def ilp(points,sample_size):
    # sample_size=100
    num_a=len(points)
    num_b=num_a*(num_a-1)//2
    m=num_b*4+num_a
    # m=num_b*4
    n=num_b+num_a
    # G=[[0 for col in range(n)] for row in range(m)]
    g_row=np.repeat([0],n)
    G=np.repeat([g_row],m,axis=0)    
    A=[0 for row in range(n)]
    h=[0 for i in range(m)]
    # start=time()
    a_combinations=[]
    for i in range(num_a-1):
        for j in range(i+1,num_a):
            a_combinations.append((i,j))
    c_b=[proximity(points[a_combinations[i][0]],points[a_combinations[i][1]]) for i in range(num_b)]
    # print(c_b)
    # end=time()
    # print(end-start)
    # exit(0)
    c_a=[0 for i in range(num_a)]
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
    for i in range(num_a):
        row=i+4*num_b
        G[row][i+num_b]=-1

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
    # return points[list(set(samples))]
    return samples_list
def get_obj(samples,size):
    obj=0
    for i in range(size-1):
        for j in range(i+1,size):
            # obj+=prox.run(samples[i],samples[j])    
            obj+=proximity(samples[i],samples[j])
    return np.around(obj,2)
def main():
    # pass
    df=pd.read_csv('data/Data/000/Trajectory/20081023025304.plt',sep=',',names=['Latitude','Longitude','0','1','2','3','4'],skiprows=6)
    # tmp=df.loc[:3,['Longitude','Latitude']].values.tolist()
    # print(tmp)
    point_set=np.array(df.loc[:39,['Longitude','Latitude']].values.tolist())
    sample_size=10
    start=time()
    # set_epsilon(point_set)
    # print(epsilon)
    ilp_samples=ilp(point_set,[sample_size])
    np.save(file='ilp_40.npy',arr=ilp_samples)

    end=time()
    print('time: {} min {} s'.format((end-start)//60,(end-start)%60))
    print(point_set.shape)
    # print(epsilon)
    # print(get_obj(ilp_samples[0],sample_size))

if __name__=='__main__':
    main()