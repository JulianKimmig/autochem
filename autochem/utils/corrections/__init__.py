import numpy as np
def norm_nm(x,y,n,m):
    if n is None:
        n=np.min(y)
    if m is None:
        m=np.max(y)
    _m=(m-n)/(y.max()-y.min())
    _c=-(y.min()*_m) + n
    return x,y*_m +_c, [_m,_c]

def norm_n1(x,n):
    return norm_nm(x,n,1)

def norm_01(x,y):
    return norm_nm(x,y,0,1)

def norm_data(x,y):
    return norm_01(x,y)


def shift_data(x,y,n):
    return x-n,y, [1,-n]

def scale_data(x,y,n):
    return x,y*n,[n,0]