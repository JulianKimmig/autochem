import unittest
import os

import matplotlib.pyplot as plt
import numpy as np

from autochem.spectra.nmr.reader import read_nmr
from autochem.spectra.nmr.utils import zoom
from autochem.utils.corrections import norm_data
from autochem.utils.corrections.baseline import rubberband_correction
from autochem.utils.signals.peak_detection import find_peaks

PLOT=True



class AbstractNMRtester:

    def _open(self):
        data, udict =read_nmr(self.TESTFILE, preprocess=True)
        return {"open":{"data":data, "udict":udict}}

    def _zoom(self):
        d= self._open()
        data,ppm = zoom(d["open"]["data"],d["open"]["udict"]["ppm_scale"],-5,20)
        d["zoom"]={"data":data,"ppm":ppm}
        return d

    def _rubberband(self):
        d = self._zoom()
        dc,corr_data = rubberband_correction(d["zoom"]["ppm"],d["zoom"]["data"])
        d["rubberband"]={"data":dc,"corr_data":corr_data}
        return d

    def _norm(self):
        d = self._rubberband()
        ndata,normed = norm_data(d["rubberband"]["data"])
        d["norm"]={"data":ndata,"normed":normed}
        return d

    def _raw_peaks(self):
        d = self._norm()
        peaks,peak_data = find_peaks(x=d["zoom"]["ppm"],y=d["norm"]["data"],min_peak_height=0.01,rel_height=0.001)
        d["raw_peaks"]={"peaks":peaks,"peak_data":peak_data}
        return d

    def _zoom_peaks(self):
        d= self._raw_peaks()
        data,ppm = zoom(
            d["norm"]["data"],
            d["zoom"]["ppm"],
            d["zoom"]["ppm"][np.floor(d["raw_peaks"]["peak_data"]["left_ips"].min()).astype(int)],
            d["zoom"]["ppm"][np.ceil(d["raw_peaks"]["peak_data"]["right_ips"].max()).astype(int)],
        )
        d["zoom_peaks"]={"data":data,"ppm":ppm}
        return d

    def _fine_peaks(self):
        d = self._zoom_peaks()
        peaks,peak_data = find_peaks(x=d["zoom_peaks"]["ppm"],y=d["zoom_peaks"]["data"],min_peak_height=0.1,rel_height=0.01,min_width=0.2)
        d["fine_peaks"]={"peaks":peaks,"peak_data":peak_data}
        return d

    def test_01_open(self):
        d=self._open()
        data, udict = d["open"]["data"],d["open"]["udict"]
        if PLOT:
            plt.plot(udict["ppm_scale"],data,label="data")
            plt.title("open")
            plt.legend()
            plt.show()
            plt.close()

    def test_02_zoom(self):
        d = self._zoom()
        if PLOT:
            plt.plot(d["zoom"]["ppm"],d["zoom"]["data"],label="zoomed data")
            plt.title("zoom")
            plt.legend()
            plt.show()
            plt.close()

    def test_03_rubberband(self):
        d = self._rubberband()
        ppm=d["zoom"]["ppm"]
        corr_data=d["rubberband"]["corr_data"]
        if PLOT:
            plt.plot(ppm,d["zoom"]["data"],label="data")
            plt.plot(ppm[corr_data['points']],d["rubberband"]["data"][corr_data['points']],"o",label="rubberband points")
            plt.plot(ppm,d["rubberband"]["data"],label="processed data")
            plt.title("rubberband")
            plt.legend()
            plt.show()
            plt.close()

    def test_04_norm(self):
        d = self._norm()
        ppm=d["zoom"]["ppm"]
        data=d["norm"]["data"]
        if PLOT:
            plt.plot(ppm,data,label="normed data")
            plt.title("norm")
            plt.legend()
            plt.show()
            plt.close()

    def test_05_peaks(self):
        d=self._raw_peaks()
        if PLOT:
            y=d["zoom"]["ppm"]
            x=d["norm"]["data"]
            plt.plot(y,x,label="normed data")
            print(d["raw_peaks"]["peak_data"])
            plt.plot(y[d["raw_peaks"]["peaks"]],x[d["raw_peaks"]["peaks"]],"o",label="peaks")
            plt.plot(y[np.floor(d["raw_peaks"]["peak_data"]['left_ips']).astype(int)],x[np.floor(d["raw_peaks"]["peak_data"]['left_ips']).astype(int)],"o",label="left_ips")
            plt.plot(y[np.floor(d["raw_peaks"]["peak_data"]['right_ips']).astype(int)],x[np.floor(d["raw_peaks"]["peak_data"]['right_ips']).astype(int)],"+",label="right_ips")
            plt.title("peaks")
            plt.legend()
            plt.show()
            plt.close()

    def test_06_zoom_peaks(self):
        d=self._zoom_peaks()
        if PLOT:
            y=d["zoom_peaks"]["ppm"]
            x=d["zoom_peaks"]["data"]
            plt.plot(y,x,label="data")
            plt.title("zoom_peaks")
            plt.legend()
            plt.show()
            plt.close()

    def test_07_fine_peaks(self):
        d=self._fine_peaks()
        if PLOT:
            y=d["zoom_peaks"]["ppm"]
            x=d["zoom_peaks"]["data"]
            plt.plot(y,x,label="data")
            plt.plot(y[d["fine_peaks"]["peaks"]],x[d["fine_peaks"]["peaks"]],"o",label="peaks")
            plt.plot(y[np.floor(d["fine_peaks"]["peak_data"]['peak_left_border']).astype(int)],x[np.floor(d["fine_peaks"]["peak_data"]['peak_left_border']).astype(int)],"*",label="left_ips")
            plt.plot(y[np.floor(d["fine_peaks"]["peak_data"]['peak_right_border']).astype(int)],x[np.floor(d["fine_peaks"]["peak_data"]['peak_right_border']).astype(int)],"+",label="right_ips")
            plt.title("fine_peaks")
            plt.legend()
            plt.show()
            plt.close()





class TestNMRMakgritek(AbstractNMRtester,unittest.TestCase):
    TESTFILE=os.path.join(os.path.dirname(__file__),"testdata","nmr","magritek_1")

#class TestNMRBruker(AbstractNMRtester,unittest.TestCase):
#    TESTFILE=
#        pass