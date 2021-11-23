import numpy as np
from scipy.optimize import minimize


def get_signal_shift(peaks,expected_peaks,allow_stretch=False):
    def mxpc(mc,src,trg):
        m , c = mc
        src = src * m + c
        src = src.flatten()
        trg = trg.flatten()

        dist_matrix = np.abs( np.subtract.outer(src,trg))
        dist=0

        while not np.isnan(dist_matrix).all():
            minidx = np.unravel_index(np.nanargmin(dist_matrix), dist_matrix.shape)
            dist += dist_matrix[minidx]
            dist_matrix[minidx[0],:]=np.nan
            dist_matrix[:,minidx[1]]=np.nan


        return dist

    bounds=[(-np.inf,np.inf),(-np.inf,np.inf)]
    if not allow_stretch:
        bounds[0]=(1,1)

    bs=[
        np.array([1, 0]),
        np.array([1, -peaks.mean()+expected_peaks.mean()]),
        np.array([1, -peaks.min()+expected_peaks.min()]),
        np.array([1, -peaks.max()+expected_peaks.max()])
    ]
    r=np.array([0])
    for b in bs:
        r = np.array([mxpc(b,peaks,expected_peaks)])


    min = minimize(mxpc,
                   bs[r.argmin()],
                   (peaks,expected_peaks),
                   method = 'Nelder-Mead',
                   bounds=bounds
                   )
    return min["x"]
