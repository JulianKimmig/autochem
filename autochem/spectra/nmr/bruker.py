import nmrglue as ng
import numpy as np
import os


def read_bruker(path):
    try:
        dic, data = ng.bruker.read_pdata(os.path.join(path, "pdata", "1"))
    except:
        raise Exception("only processed readable so far")
    #        dic,data = ng.bruker.read(path)

    udic = ng.bruker.guess_udic(dic, data)

    udic["dic"] = dic
    udic["datatype"] = "bruker"
    udic[0]["complex"] = np.iscomplex(data).any()

    return udic, data
