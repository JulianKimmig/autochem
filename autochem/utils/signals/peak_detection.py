import numpy as np
from scipy import signal


def find_peaks(y,min_peak_height=0.0,min_width=0,x=None,rel_height=.01):
    if x is not None:
        points_per_x = len(x)/(x.max()-x.min())
        min_width=points_per_x*min_width
    peaks,peak_data = signal.find_peaks(y, height=min_peak_height,
                                     #   prominence=MIN_PEAK_HEIGHT,
                                        width=min_width,
                                        rel_height=1-rel_height,
                                        )

    min_border_index=np.zeros_like(peaks)
    for i in range(1,len(peaks)):
        min_border_index[i]=y[peaks[i-1]:peaks[i]].argmin()+peaks[i-1]

    max_border_index=np.zeros_like(peaks)
    for i in range(len(peaks)-1):
        max_border_index[i]=y[peaks[i]:peaks[i+1]].argmin()+peaks[i]
    max_border_index[-1]=len(y)-1


    data_indices=np.arange(len(y))

    calc_int_bord_left=np.zeros_like(peaks)
    for i,p in enumerate(peaks):
        a = (y[:p]<=peak_data['peak_heights'][i]*rel_height)
        calc_int_bord_left[i]=data_indices[:p][a].max()

    calc_int_bord_right=np.zeros_like(peaks)
    for i,p in enumerate(peaks):
        a = (y[p:]<=peak_data['peak_heights'][i]*rel_height)
        calc_int_bord_right[i]=data_indices[p:][a].min()


    peak_left_border=np.maximum(
        #np.floor(peak_data['left_ips']).astype(int),
        calc_int_bord_left,
        min_border_index,
    )
    peak_right_border=np.minimum(
        #    np.ceil(peak_data['right_ips']).astype(int),
        calc_int_bord_right,
        max_border_index,
    )

    peak_data["peak_left_border"]=peak_left_border
    peak_data["peak_right_border"]=peak_right_border
    return peaks,peak_data