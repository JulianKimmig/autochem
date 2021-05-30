import nmrglue as ng
import numpy as np
import os


def read_magritek(path):
    dic, raw_data = ng.jcampdx.read(os.path.join(path, "nmr_fid.dx"))
    udic = ng.jcampdx.guess_udic(dic, raw_data)
    npoints = int(udic[0]["size"])
    data = np.empty((npoints,), dtype="complex128")
    data.real = raw_data[0][:]
    data.imag = raw_data[1][:]

    udic["acqu"] = {}
    with open(path + "/acqu.par") as f:
        for line in f:
            eq_index = line.find("=")
            var_name = line[:eq_index].strip()
            number = line[eq_index + 1 :].strip()
            if (number.startswith('"') or number.startswith('"')) and (
                number.endswith('"') or number.endswith('"')
            ):
                number = number[1:-1]
            elif "." in number:
                try:
                    number = float(number)
                except:
                    pass
            else:
                try:
                    number = int(number)
                except:
                    pass

            udic["acqu"][var_name] = number

    udic[0]["sw"] = int(udic["acqu"]["bandwidth"])

    # sometimes in Hz somtimes in kHz...
    if udic[0]["sw"] < 1000:
        udic[0]["sw"] *= 1000

    udic[0]["car"] = float(udic["acqu"]["lowestFrequency"]) + udic[0]["sw"] / 2

    udic["datatype"] = "magritek"
    udic["dic"] = dic
    udic[0]["complex"] = np.iscomplex(data).any()
    return udic, data
