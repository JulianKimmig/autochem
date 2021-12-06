def norm_nm(x,n,m):
    _m=(m-n)/(x.max()-x.min())
    _c=-(x.min()*_m) + n
    return x*_m +_c, [_m,_c]

def norm_n1(x,n):
    return norm_nm(x,n,1)

def norm_01(x):
    return norm_nm(x,0,1)

def norm_data(x):
    return norm_01(x)


def shift_data(x,n):
    return x-n, [1,-n]

def scale_data(x,n):
    return x*n,[n,0]