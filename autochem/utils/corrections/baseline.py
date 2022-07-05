from statistics import median
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull

from autochem.utils.data_norm import sort_xy


def rubberband_correction(x, y):
    x, y = sort_xy(x, y)
    v = ConvexHull(np.array(list(zip(x, y)))).vertices
    v = np.roll(v, -v.argmin())
    v = v[: v.argmax()+1]
    bl=np.interp(x, x[v], y[v])
    return y - bl, {
        "points": v,
        "baseline": bl
    }




def asymmetric_least_squares_smoothing(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = np.zeros_like(y)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y-z, {
        "baseline": z,
    }

def median_correction(data):
    median=np.median(data)
    print(median)
    return data-median, {"median": median}