from autochem.spectra.nmr.bruker import read_bruker
from autochem.spectra.nmr.magritek import read_magritek
from autochem.spectra.nmr.utils import process_nmr_signal, get_ppm_scale, sort_ppm


class NMRReadError(Exception):
    pass


def _try_read_nmr(path, type):
    # print("read",path,"as",type)
    try:
        if type == "magritek":
            return read_magritek(path)
        elif type == "bruker":
            return read_bruker(path)
        else:
            return None
    except Exception as e:
        return None


def read_nmr(path, type=None, preprocess=True):
    if type is None:
        type = ["magritek", "bruker"]
    if not isinstance(type, (list, tuple)):
        type = [type]

    udict = data = None
    for t in type:
        r = _try_read_nmr(path, t)
        if r is None:
            continue
        else:
            udict, data = r
            break

    if udict is None:
        raise NMRReadError("cannot read nmr")

    if preprocess:
        udict["ppm_scale"] = get_ppm_scale(udict)
        if udict[0]["complex"]:
            udict["raw_data"] = data
            data = process_nmr_signal(data)
        udict["ppm_scale"], data = sort_ppm(udict["ppm_scale"], data)
    return data, udict
