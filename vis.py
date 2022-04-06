import signal
import time
from utils import *
import numpy as np
# pool=[]

def handler(signum, stack):
    # print ('Alarm: {}'.format(time.ctime()) )
    # print(len(pool))
    print(stack)
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
    def run(self,point_set,k,timeout,stop_points):
        # global pool
        pool=[]
        i=0
        idx=0
        set_size=point_set.shape[0]
        while i<stop_points:
            i+=1
            start=time.time()
            # for point in point_set:
            while idx<set_size:
                point=point_set[idx]
                if time.time()-start>timeout:
                    # print(time.ctime())
                    break
                if len(pool)<k:
                    pool=self.expand(pool,point)
                else:
                    pool=self.expand(pool,point)
                    pool=self.shrink(pool)
                idx+=1
            print(idx)
            s=[point[0] for point in pool] #(coordinates,responsibility) for each point in r
            np.save('stop_points/int_{}_{}.npy'.format(set_size,i*timeout),s)
            print('save {}th file'.format(i))
        return np.array(s)
import os
import pandas as pd
plt_root_path='data/Data/000/Trajectory/'
threshold=10000
df=pd.read_csv('data/Data/000/Trajectory/20081023025304.plt',sep=',',names=['Latitude','Longitude','0','1','2','3','4'],skiprows=6)

point_set=np.array(df.loc[:,['Longitude','Latitude']].values.tolist())
point_set_size=point_set.shape[0]

for file in os.listdir(plt_root_path)[1:]:
    # print(file)
    df=pd.read_csv(plt_root_path+file,sep=',',names=['Latitude','Longitude','0','1','2','3','4'],skiprows=6)
    df_points=np.array(df.loc[:,['Longitude','Latitude']].values.tolist())
    point_set=np.concatenate((point_set,df_points))
    point_set_size+=df_points.shape[0]
    if point_set_size>threshold:
        break
# signal.signal(signal.SIGALRM, handler)
# signal.alarm(2)
prox=proximity(point_set,set_eps=True)
int_generator=interchange(prox)
int_samples=int_generator.run(point_set,500,10,5)
print(int_samples.shape)
# time.sleep(4)
# print ("interrupted in {}".format(time.ctime()))