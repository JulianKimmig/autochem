import numpy as np
from scipy.spatial import ConvexHull

from autochem.utils.data_norm import sort_xy


def rubberband_correction(x, y):
    x, y = sort_xy(x, y)
    v = ConvexHull(np.array(list(zip(x, y)))).vertices
    v = np.roll(v, -v.argmin())
    v = v[: v.argmax()]
    return y - np.interp(x, x[v], y[v]), {
        "points": v,
        "baseline": np.interp(x, x[v], y[v]),
    }
