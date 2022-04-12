import numpy as np
class proximity:
    def __init__(self,point_set,set_eps=False) -> None:
        if set_eps==False:
            self.epsilon=3.2147064340016937e-07
            # self.epsilon=3.797682913998397e-07 #2000
            # self.epsilon=1.8988414569991985e-07
        else:
            self.epsilon=self.set_epsilon(point_set)
    def set_epsilon(self,point_set):
        max_norm=0
        i=0
        point_set_size=point_set.shape[0]
        while i <point_set_size-1:
            j=i+1
            while j<point_set_size-1:
                norm=np.linalg.norm(point_set[i]-point_set[j])
                if max_norm<norm:
                    max_norm=norm  
                j+=1
            i+=1
        # return max_norm/100
        return np.power(max_norm/100,2)*2
        # return np.power(max_norm/100,2)
    def run(self,point_1,point_2):
        return np.exp(-np.power(np.linalg.norm(point_1-point_2),2)/self.epsilon)    
        # return     np.linalg.norm(point_1-point_2)/self.epsilon

    # def proximity(point_1,point_2):
    #     try:
    #         if np.sum(np.where((point_1==point_2),0,1))==0:
    #             return 0
    #         return np.exp(-np.power(np.linalg.norm(point_1-point_2),2)/epsilon)
    #     except:
    #         print(point_1,point_2)
    #         exit(0)
def get_obj(samples,prox):
    obj=0
    size=samples.shape[0]
    for i in range(size-1):
        for j in range(i+1,size):
            obj+=prox.run(samples[i],samples[j])    
            # obj+=proximity(samples[i],samples[j])
    return np.around(obj,2)