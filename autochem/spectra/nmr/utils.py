import nmrglue as ng
import numpy as np

from autochem.utils.data_norm import sort_xy


def get_ppm_scale(udic):
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()
    return ppm_scale


def process_nmr_signal(data):
    p_data = ng.proc_base.fft(data)  # fourier trans
    p_data = ng.proc_base.di(p_data)  # remove imaginary pat
    return p_data


def sort_ppm(ppm_scale, data):
    return sort_xy(ppm_scale, data)


def zoom(data, ppm_scale, xmin, xmax):
    ppm_scale, data = sort_ppm(ppm_scale, data)
    idx_min = (np.abs(ppm_scale - xmin)).argmin()
    idx_max = (np.abs(ppm_scale - xmax)).argmin()
    return data[idx_min:idx_max+1], ppm_scale[idx_min:idx_max+1],[int(idx_min),int(idx_max)]
