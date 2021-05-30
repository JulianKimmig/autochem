import numpy as np
from scipy import signal
from scipy.integrate import simps


def find_peaks(y, x=None, min_peak_height=0.0, min_distance=0, rel_height=.01, rel_prominence=0, center="max",max_width=np.inf, **kwargs):
    if x is not None:

        points_per_x = len(x) / (x.max() - x.min())
        min_distance = points_per_x * min_distance
        max_width=points_per_x * max_width

    if "prominence" not in kwargs:
        kwargs["prominence"] = 0
    if "width" not in kwargs:
        kwargs["width"] = 0
    peaks, peak_data = signal.find_peaks(y, height=min_peak_height,
                                         #   prominence=MIN_PEAK_HEIGHT,
                                         distance=max(1,min_distance),
                                         rel_height=1 - rel_height,
                                         **kwargs
                                         )

    rel_prom_mask = (peak_data['prominences'] / peak_data['peak_heights']) >= rel_prominence
    peaks = peaks[rel_prom_mask]

    for k in list(peak_data.keys()):
        peak_data[k] = peak_data[k][rel_prom_mask]

    min_border_index = np.zeros_like(peaks)
    for i in range(1, len(peaks)):
        min_border_index[i] = y[peaks[i - 1]:peaks[i]].argmin() + peaks[i - 1]

    max_border_index = np.zeros_like(peaks)
    for i in range(len(peaks) - 1):
        max_border_index[i] = y[peaks[i]:peaks[i + 1]].argmin() + peaks[i]
    max_border_index[-1] = len(y) - 1

    data_indices = np.arange(len(y))

    calc_int_bord_left = np.zeros_like(peaks)
    for i, p in enumerate(peaks):
        a = (y[:p] <= peak_data['peak_heights'][i] * rel_height)
        a[:int(max(1,p-np.ceil(max_width/2)))]=True

        calc_int_bord_left[i] = data_indices[:p][a].max()

    calc_int_bord_right = np.zeros_like(peaks)
    for i, p in enumerate(peaks):
        a = (y[p:] <= peak_data['peak_heights'][i] * rel_height)
        a[int(min(a.shape[0]-1,np.ceil(max_width/2))):]=True
        calc_int_bord_right[i] = data_indices[p:][a].min()

    peak_left_border = np.maximum(
        # np.floor(peak_data['left_ips']).astype(int),
        calc_int_bord_left,
        min_border_index,
    )
    peak_right_border = np.minimum(
        #    np.ceil(peak_data['right_ips']).astype(int),
        calc_int_bord_right,
        max_border_index,
    )

    peak_data["peak_left_border"] = np.floor(peak_left_border).astype(int)
    peak_data["peak_right_border"] = np.ceil(peak_right_border).astype(int)

    if center == "max":
        pass
    elif center == "mean":
        peaks = (peak_data["peak_left_border"] + peak_data["peak_right_border"]) / 2
        peaks = np.round(peaks).astype(int)
    elif center == "median":
        ms = []
        for i in range(len(peaks)):
            cs = y[peak_data["peak_left_border"][i]:peak_data["peak_right_border"][i]].cumsum()
            cs = cs >= cs[-1] / 2
            cs = peak_data["peak_left_border"][i] + cs.argmax()
            ms.append(cs)
        peaks = np.array(ms)
    #print(peak_data)
    return peaks, peak_data


from scipy import integrate

def peak_integration(x, y, peaks=None, peak_data=None, **kwargs):
    """

    :type peaks: object
    """
    if peaks is None or peak_data is None or len(kwargs) > 0:
        peaks, peak_data = find_peaks(y, x, **kwargs)
    #integrals = np.zeros_like(peaks, dtype=float)
    #for i, p in enumerate(peaks):
    #    integrals[i] = simps(y[peak_data["peak_left_border"][i]:peak_data["peak_right_border"][i]],
    #                          x[peak_data["peak_left_border"][i]:peak_data["peak_right_border"][i]])
    #peak_data["integrals"] = integrals
    cum_i = integrate.cumtrapz(y, x, initial=0)
    peak_data["cum_integral"] = cum_i

    from_cumintegrals = np.zeros_like(peaks, dtype=float)
    for i, p in enumerate(peaks):
        from_cumintegrals[i] =cum_i[peak_data["peak_right_border"][i]]-cum_i[peak_data["peak_left_border"][i]]

    peak_data["integrals"]=from_cumintegrals
    return peaks, peak_data


class PeakNotFoundError(Exception):
    pass

def get_reference_peak(peaks,target,max_diff=np.inf):
    pidx=np.argmin(np.abs(peaks-target))
    peak=peaks[pidx]
    if np.abs(peak-target)>max_diff:
        raise PeakNotFoundError("No peak close to {}, closest peak is {}".format(target,peak))
    return pidx,peak