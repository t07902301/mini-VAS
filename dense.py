import numpy as np
import heapq
import pandas as pd
from utils import *
from time import time

adj_mat=[]
def get_adj_mat(point_set,k,prox):
    point_set_size=len(point_set)
    max_prox=0
    i=0
    while i <point_set_size-1:
        row=[]
        j=i+1
        while j<point_set_size-1-i:
            prox_value=prox.run(point_set[i],point_set[j])
            # if max_prox<prox:
            #     max_prox=prox
            row.append((prox_value,(i,j)))
            j+=1
        i+=1
        row.sort(key=lambda tup:tup[0])#ascending
        adj_mat.append(row[:k])
def get_adj_mat_na(point_set):
    '''
    Without heaps
    '''
    adj_mat_na=[]
    point_set_size=len(point_set)
    i=0
    while i <point_set_size-1:
        row=[]
        j=0
        while j<point_set_size-1:
            prox=proximity(point_set[i],point_set[j])
            row.append(prox)
            j+=1
        i+=1
        adj_mat_na.append(row)
    return np.array(adj_mat_na)
def find_min_adj(adj_mat_na,point_set_size):
    min_pair=(0,0)
    min_key=1.5
    i=0
    j=0
    while i <point_set_size-1:
        j=i+1
        while j<point_set_size-1:
            try:
                if min_key>adj_mat_na[i][j]:
                    min_key=adj_mat_na[i][j]
                    min_pair=(i,j)
            except:
                print(adj_mat_na.shape)
                print(i,j)
                exit(0)
            j+=1
        i+=1
    return min_pair
def update_mat(adj_mat,node_pair):
    n1,n2=node_pair
    # adj_mat=np.delete(adj_mat,n1,0)
    # adj_mat=np.delete(adj_mat,n2,1)
    adj_mat[n1,:]=np.ones(adj_mat.shape[1])
    adj_mat[:,n2]=np.ones(adj_mat.shape[0])
    # print(adj_mat.shape)
    return adj_mat
def dense_k_na(adj_mat,k):
    '''
    Without Heaps
    '''
    subgraph=[]
    i=0
    while i<k//2:
        min_pair=find_min_adj(adj_mat,adj_mat.shape[0])
        subgraph+=list(min_pair)
        adj_mat=update_mat(adj_mat,min_pair)
        i+=1
    return subgraph

#TODO max_prox fixed?      
heaps=[]
def get_heaps():
    for row in adj_mat:
        # negative_row=[(tup[0]-max_prox,tup[1]) for tup in row]
        negative_row=[(tup[0],tup[1]) for tup in row]
        heapq.heapify(negative_row)
        heaps.append(negative_row)

def find_min_heap():
    min_key=1.5
    min_idx=0
    min_pair=(0,0)
    for idx,heap in enumerate(heaps):
        # print(heap)
        if len(heap)==0:
            continue
        key,pair=heap[0]
        if min_key>key:
            min_idx=idx
            min_pair=pair
            min_key=key       
    return min_pair,min_idx
def update_heaps(closed_points):
    for heap in heaps:
        while True and len(heap)>0:
            min_key,min_pair=heap[0]
            # print(min_pair)
            if closed_points.intersection(set(min_pair)):
                heapq.heappop(heap)
            else:
                break
        
def dense_k(k):
    global heaps
    global adj_mat
    subgraph=set()
    i=0
    while i<k//2:
        if len(heaps)==0:
            break        
        min_pair,min_idx=find_min_heap()
        # print(heaps)
        del(heaps[min_idx])
        new_points=set(min_pair)
        subgraph=subgraph.union(new_points)
        update_heaps(subgraph)
        i+=1
    if k%2==1:
        subgraph.union()
    heaps=[] # clear heaps
    adj_mat=[] #clear the ajacent matrix
    return subgraph
if __name__=='__main__':
    df=pd.read_csv('data/Data/000/Trajectory/20081023025304.plt',sep=',',names=['Latitude','Longitude','0','1','2','3','4'],skiprows=6)
    point_set=np.array(df.loc[:49,['Longitude','Latitude']].values.tolist())
    # exit(0)
    prox=proximity(point_set)
    start=time()
    k=10
    get_adj_mat(point_set,k,prox)
    get_heaps()
    subgraph=dense_k(k)

    # adj_mat=get_adj_mat_na(point_set)
    # subgraph=dense_k_na(adj_mat,k)

    end=time()
    # print('time: {} min {} s'.format((end-start)//60,(end-start)%60))
    print('running time: ',time()-start)
    # print(list(subgraph))
    print(get_obj(point_set[list(subgraph)],prox))
    # print(subgraph)

    # get_adj_mat(point_set,sample_size)
    # get_heaps()
    # subgraph=dense_k(sample_size)
    # dk_samples=point_set[list(subgraph)]
    # dk_x=dk_samples[:,0]
    # dk_y=dk_samples[:,1]
    # plot.scatter(dk_x,dk_y,c='b')






    

    