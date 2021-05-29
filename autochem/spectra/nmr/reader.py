from autochem.spectra.nmr.magritek import read_magritek
from autochem.spectra.nmr.utils import process_nmr_signal, get_ppm_scale


def read_nmr(path,type=None,preprocess=True):
    try:
        udict,data = read_magritek(path)
    except:
        raise ValueError("cannot read nmr")

    if preprocess:
        udict["raw_data"]=data
        udict["ppm_scale"]=get_ppm_scale(udict)
        data=process_nmr_signal(data)
    return data, udict