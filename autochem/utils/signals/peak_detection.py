from logging import warning

import numpy as np
from scipy import signal

PEAK_INDEX_VALUES=['left_bases', 'right_bases', 'left_ips', 'right_ips', "peak_left_border", "peak_right_border",
                      "peak_maximum", "peak_mean", "peak_median"]

PEAK_SCALE_DEPENDING_VALUES=['peak_heights', 'prominences', 'width_heights', 'integrals','cum_integral']

def cut_peaks_data(peaks,peak_data,idx1,idx2):
    idx1,idx2 = min(idx1,idx2),max(idx1,idx2)

    peak_mask=(peaks>=idx1) & (peaks<=idx2)
    peaks=peaks[peak_mask] - idx1

    for ys in PEAK_INDEX_VALUES:
        if ys in peak_data:
            peak_data[ys] = peak_data[ys][peak_mask]- idx1



    return peaks,peak_data


def factorize_peak_data(peak_data,scale_factor=1):
    for ys in PEAK_SCALE_DEPENDING_VALUES:
        if ys in peak_data:
            peak_data[ys] *= scale_factor

    return peak_data

def merge_peaks_data(peaks1, peaks2, peaks_data1, peaks_data2, x1, x2, peaks1_y_shift=0, peaks2_y_shift=0,
                     peaks1_norm_fac=1,
                     peaks2_norm_fac=1):
    peaks1 += peaks1_y_shift
    peaks2 += peaks2_y_shift
    peaks = np.concatenate([peaks2, peaks1])
    _, idx = np.unique(peaks, return_index=True)
    peaks = peaks[idx]

    peaks_data = {}
    common_keys = set(peaks_data1).intersection(peaks_data2)
    y_shifted = PEAK_INDEX_VALUES.copy()

    scale_factored = ['peak_heights', 'prominences', 'width_heights', 'integrals']
    manually = ['cum_integral']
    independend = ['widths']
    x=None
    for ys in y_shifted:
        if ys in common_keys:
            peaks_data[ys] = np.concatenate([
                peaks_data2[ys] + peaks2_y_shift,
                peaks_data1[ys] + peaks1_y_shift,
            ])[idx]

    for sf in scale_factored:
        if sf in common_keys:
            peaks_data[sf] = np.concatenate([
                peaks_data2[sf] * peaks2_norm_fac,
                peaks_data1[sf] * peaks1_norm_fac,
            ])[idx]

    for idk in independend:
        if idk in common_keys:
            peaks_data[idk] = np.concatenate([
                peaks_data2[idk],
                peaks_data1[idk],
            ])[idx]

    if "cum_integral" in common_keys:

        #
        #
        if x1.min() < x2.min() and x1.max() < x2.min():  # 111112222
            c1 = peaks_data1["cum_integral"] * peaks1_norm_fac
            c2 = peaks_data2["cum_integral"] * peaks2_norm_fac + c1[-1]
            x=np.concatenate([x1, x2])
            peaks_data["cum_integral"] = np.concatenate([c1, c2])

        elif x2.min() < x1.min() and x2.max() < x1.min():  # 222211111
            c1 = peaks_data2["cum_integral"] * peaks2_norm_fac
            c2 = peaks_data1["cum_integral"] * peaks1_norm_fac + c1[-1]
            peaks_data["cum_integral"] = np.concatenate([c1, c2])
            x=np.concatenate([x2, x1])

        elif x1.min() < x2.min() and x1.max() < x2.min() and x1.max() < x2.max():  # 111133222
            ol_ix = (x1 >= x2.min()).argmax()
            c1 = peaks_data1["cum_integral"][:ol_ix] * peaks1_norm_fac
            c2 = peaks_data2["cum_integral"] * peaks2_norm_fac + c1[-1]
            peaks_data["cum_integral"] = np.concatenate([c1, c2])
            x=np.concatenate([x1[:ol_ix], x2])

        elif x2.min() < x1.min() and x2.max() < x1.min() and x2.max() < x1.max():  # 222233111
            ol_ix = (x2 >= x1.min()).argmax()
            c1 = peaks_data2["cum_integral"][:ol_ix] * peaks2_norm_fac
            c2 = peaks_data1["cum_integral"] * peaks1_norm_fac + c1[-1]
            peaks_data["cum_integral"] = np.concatenate([c1, c2])
            x=np.concatenate([x2[:ol_ix], x1])

        elif x1.min() < x2.min() and x1.max() > x2.max():  # 113333111
            ol_ix1 = (x1 >= x2.min()).argmax()
            ol_ix2 = (x1 > x2.max()).argmax()
            c1 = peaks_data1["cum_integral"][:ol_ix1] * peaks1_norm_fac
            c2 = peaks_data2["cum_integral"] * peaks2_norm_fac + c1[-1]
            c3 = peaks_data1["cum_integral"][ol_ix2:] * peaks1_norm_fac + c2[-1]
            peaks_data["cum_integral"] = np.concatenate([c1, c2, c3])
            x=np.concatenate([x1[:ol_ix1], x2,x1[ol_ix2:]])

        elif x2.min() < x1.min() and x2.max() > x1.max():  # 223333222
            ol_ix1 = (x2 >= x1.min()).argmax()
            ol_ix2 = (x2 > x1.max()).argmax()
            c1 = peaks_data2["cum_integral"][:ol_ix1] * peaks2_norm_fac
            c2 = peaks_data1["cum_integral"] * peaks1_norm_fac + c1[-1]
            c3 = peaks_data2["cum_integral"][ol_ix2:] * peaks2_norm_fac + c2[-1]
            peaks_data["cum_integral"] = np.concatenate([c1, c2, c3])
            x=np.concatenate([x2[:ol_ix1], x1,x2[ol_ix2:]])
        else:
            raise ValueError("???") # should not occure

    for k in common_keys:
        if k not in y_shifted + scale_factored + independend + manually:
            warning(f"found unknown peak attribure '{k}':\n{peaks_data1[k]}\n{peaks_data2[k]}")
    return peaks, peaks_data, x


def manual_peak_finder(y,x,peak_ranges,**kwargs):

    peaks,data,xm=None,None,None

    for r in peak_ranges:
        in_ppm=(x>=r[0]) & (x<=r[1])
        p,d = find_peaks(y[in_ppm],
                         x[in_ppm],
                         min_distance=r[1]-r[0],
                         **kwargs)
        d["peak_left_border"][:]=0
        d["peak_right_border"][:]=in_ppm.sum()-1
        if peaks is not None:
            peaks,data,_ = merge_peaks_data(
                peaks,p,
                peaks_data1=data,
                peaks_data2=d,
                peaks1_y_shift=xm,
                peaks2_y_shift=in_ppm.argmax(),
                x1=None,
                x2=None

            )
            xm=0
        else:
            peaks,data = p,d
            xm = in_ppm.argmax()

    return peaks,data

def find_peaks(
        y,
        x=None,
        min_peak_height=0.0,
        min_distance=0,
        rel_height=0.01,
        rel_prominence=0,
        center="max",
        max_width=np.inf,
        **kwargs
):
    if x is not None:
        points_per_x = len(x) / (x.max() - x.min())
        min_distance = points_per_x * min_distance
        max_width = points_per_x * max_width

    if "prominence" not in kwargs:
        kwargs["prominence"] = 0
    if "width" not in kwargs:
        kwargs["width"] = 0
    peaks, peak_data = signal.find_peaks(
        y,
        height=min_peak_height,
        #   prominence=MIN_PEAK_HEIGHT,
        distance=max(1, min_distance),
        rel_height=1 - rel_height,
        **kwargs
    )

    rel_prom_mask = (
                            peak_data["prominences"] / peak_data["peak_heights"]
                    ) >= rel_prominence
    peaks = peaks[rel_prom_mask]


    for k in list(peak_data.keys()):
        peak_data[k] = peak_data[k][rel_prom_mask]

    min_border_index = np.zeros_like(peaks)
    for i in range(1, len(peaks)):
        min_border_index[i] = y[peaks[i - 1]: peaks[i]].argmin() + peaks[i - 1]

    max_border_index = np.zeros_like(peaks)
    for i in range(len(peaks) - 1):
        max_border_index[i] = y[peaks[i]: peaks[i + 1]].argmin() + peaks[i]
    max_border_index[-1] = len(y) - 1

    data_indices = np.arange(len(y))

    calc_int_bord_left = np.zeros_like(peaks)
    for i, p in enumerate(peaks):
        a = y[:p] <= peak_data["peak_heights"][i] * rel_height
        a[: int(max(1, p - np.ceil(max_width / 2)))] = True

        calc_int_bord_left[i] = data_indices[:p][a].max()

    calc_int_bord_right = np.zeros_like(peaks)
    for i, p in enumerate(peaks):
        a = y[p:] <= peak_data["peak_heights"][i] * rel_height
        a[int(min(a.shape[0] - 1, np.ceil(max_width / 2))):] = True
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

    peak_data["peak_maximum"] = peaks.copy()
    peak_data["peak_mean"] = np.round((peak_data["peak_left_border"] + peak_data["peak_right_border"]) / 2).astype(int)
    ms = []
    for i in range(len(peaks)):
        cs = y[
             peak_data["peak_left_border"][i]: peak_data["peak_right_border"][i]
             ].cumsum()
        cs = cs >= cs[-1] / 2
        cs = peak_data["peak_left_border"][i] + cs.argmax()
        ms.append(cs)
    peak_data["peak_median"] = np.array(ms)

    if center == "max":
        pass
    elif center == "mean":
        peaks = peak_data["peak_mean"].copy()
    elif center == "median":
        peaks = peak_data["peak_median"].copy()
    # print(peak_data)
    return peaks, peak_data


from scipy import integrate

def peak_integration(x, y, peaks=None, peak_data=None, **kwargs):
    """

    :type peaks: object
    """
    if peaks is None or peak_data is None or len(kwargs) > 0:
        peaks, peak_data = find_peaks(y, x, **kwargs)
    # integrals = np.zeros_like(peaks, dtype=float)
    # for i, p in enumerate(peaks):
    #    integrals[i] = simps(y[peak_data["peak_left_border"][i]:peak_data["peak_right_border"][i]],
    #                          x[peak_data["peak_left_border"][i]:peak_data["peak_right_border"][i]])
    # peak_data["integrals"] = integrals
    cum_i = integrate.cumtrapz(y, x, initial=0)
    peak_data["cum_integral"] = cum_i

    from_cumintegrals = np.zeros_like(peaks, dtype=float)
    for i, p in enumerate(peaks):
        from_cumintegrals[i] = (
                cum_i[peak_data["peak_right_border"][i]]
                - cum_i[peak_data["peak_left_border"][i]]
        )

    peak_data["integrals"] = from_cumintegrals
    return peaks, peak_data


class PeakNotFoundError(Exception):
    pass


def get_reference_peak(peaks, target, max_diff=np.inf):
    pidx = np.argmin(np.abs(peaks - target))
    peak = peaks[pidx]
    if np.abs(peak - target) > max_diff:
        raise PeakNotFoundError(
            "No peak close to {}, closest peak is {}".format(target, peak)
        )
    return pidx, peak
