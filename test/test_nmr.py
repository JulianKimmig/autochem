import unittest
import os

import matplotlib.pyplot as plt
import numpy as np

from autochem.spectra.nmr.reader import read_nmr
from autochem.spectra.nmr.utils import zoom
from autochem.utils.corrections import norm_data
from autochem.utils.corrections.baseline import rubberband_correction
from autochem.utils.signals.peak_detection import (
    find_peaks,
    peak_integration,
    get_reference_peak,
)

PLOT = True


class AbstractNMRtester:
    def _open(self):
        data, udict = read_nmr(self.TESTFILE, preprocess=True)
        return {"open": {"data": data, "udict": udict}}

    def _zoom(self):
        d = self._open()
        data, ppm = zoom(d["open"]["data"], d["open"]["udict"]["ppm_scale"], -5, 20)
        d["zoom"] = {"data": data, "ppm": ppm}
        return d

    def _rubberband(self):
        d = self._zoom()
        dc, corr_data = rubberband_correction(d["zoom"]["ppm"], d["zoom"]["data"])
        d["rubberband"] = {"data": dc, "corr_data": corr_data}
        return d

    def _norm(self):
        d = self._rubberband()
        ndata, normed = norm_data(d["rubberband"]["data"])
        d["norm"] = {"data": ndata, "normed": normed}
        return d

    def _raw_peaks(self):
        d = self._norm()
        peaks, peak_data = find_peaks(
            y=d["norm"]["data"],
            x=d["zoom"]["ppm"],
            min_peak_height=0.05,
            rel_height=0.001,
            max_width=5,
        )
        d["raw_peaks"] = {"peaks": peaks, "peak_data": peak_data}
        return d

    def _zoom_peaks(self):
        d = self._raw_peaks()
        data, ppm = zoom(
            d["norm"]["data"],
            d["zoom"]["ppm"],
            d["zoom"]["ppm"][
                np.floor(d["raw_peaks"]["peak_data"]["left_ips"].min()).astype(int)
            ],
            d["zoom"]["ppm"][
                np.ceil(d["raw_peaks"]["peak_data"]["right_ips"].max()).astype(int)
            ],
        )
        d["zoom_peaks"] = {"data": data, "ppm": ppm}
        return d

    def _find_peaks(self):
        d = self._zoom_peaks()
        peaks, peak_data = find_peaks(
            y=d["zoom_peaks"]["data"],
            x=d["zoom_peaks"]["ppm"],
            min_peak_height=0.1,
            min_distance=0.1,
            rel_height=0.01,
            rel_prominence=0.4,
            center="median",
        )
        d["find_peaks"] = {"peaks": peaks, "peak_data": peak_data}
        return d

    def _integrate_peaks(self):
        d = self._find_peaks()
        peaks, peak_data = peak_integration(
            x=d["zoom_peaks"]["ppm"],
            y=d["zoom_peaks"]["data"],
            peaks=d["find_peaks"]["peaks"],
            peak_data=d["find_peaks"]["peak_data"],
        )
        d["integrate_peaks"] = {"peaks": peaks, "peak_data": peak_data}
        return d

    def _ref_peaks(self):
        d = self._integrate_peaks()
        ref_peak = self.REF_PEAK
        ref_norm_int = self.REF_PEAK_A
        pidx, peak = get_reference_peak(
            d["zoom_peaks"]["ppm"][d["integrate_peaks"]["peaks"]], ref_peak, max_diff=1
        )
        normf = ref_norm_int / d["integrate_peaks"]["peak_data"]["integrals"][pidx]
        d["ref_peaks"] = {
            "integrals": d["integrate_peaks"]["peak_data"]["integrals"] * normf,
            "cum_integral": d["integrate_peaks"]["peak_data"]["cum_integral"] * normf,
        }
        return d

    def test_01_open(self):
        d = self._open()
        data, udict = d["open"]["data"], d["open"]["udict"]
        if PLOT:
            plt.plot(udict["ppm_scale"], data, label="data")
            plt.title("open")
            plt.legend()
            plt.show()
            plt.close()

    def test_02_zoom(self):
        d = self._zoom()
        if PLOT:
            plt.plot(d["zoom"]["ppm"], d["zoom"]["data"], label="zoomed data")
            plt.title("zoom")
            plt.legend()
            plt.show()
            plt.close()

    def test_03_rubberband(self):
        d = self._rubberband()
        ppm = d["zoom"]["ppm"]
        corr_data = d["rubberband"]["corr_data"]
        if PLOT:
            plt.plot(ppm, d["zoom"]["data"], label="data")
            plt.plot(
                ppm[corr_data["points"]],
                d["rubberband"]["data"][corr_data["points"]],
                "o",
                label="rubberband points",
            )
            plt.plot(ppm, d["rubberband"]["data"], label="processed data")
            plt.title("rubberband")
            plt.legend()
            plt.show()
            plt.close()

    def test_04_norm(self):
        d = self._norm()
        ppm = d["zoom"]["ppm"]
        data = d["norm"]["data"]
        if PLOT:
            plt.plot(ppm, data, label="normed data")
            plt.title("norm")
            plt.legend()
            plt.show()
            plt.close()

    def test_05_peaks(self):
        d = self._raw_peaks()
        if PLOT:
            y = d["zoom"]["ppm"]
            x = d["norm"]["data"]
            plt.plot(y, x, label="normed data")
            print(d["raw_peaks"]["peak_data"])
            plt.plot(
                y[d["raw_peaks"]["peaks"]],
                x[d["raw_peaks"]["peaks"]],
                "o",
                label="peaks",
            )
            plt.plot(
                y[np.floor(d["raw_peaks"]["peak_data"]["left_ips"]).astype(int)],
                x[np.floor(d["raw_peaks"]["peak_data"]["left_ips"]).astype(int)],
                "o",
                label="left_ips",
            )
            plt.plot(
                y[np.floor(d["raw_peaks"]["peak_data"]["right_ips"]).astype(int)],
                x[np.floor(d["raw_peaks"]["peak_data"]["right_ips"]).astype(int)],
                "+",
                label="right_ips",
            )
            plt.title("peaks")
            plt.legend()
            plt.show()
            plt.close()

    def test_06_zoom_peaks(self):
        d = self._zoom_peaks()
        if PLOT:
            y = d["zoom_peaks"]["ppm"]
            x = d["zoom_peaks"]["data"]
            plt.plot(y, x, label="data")
            plt.title("zoom_peaks")
            plt.legend()
            plt.show()
            plt.close()

    def test_07_find_peaks(self):
        d = self._find_peaks()
        if PLOT:
            y = d["zoom_peaks"]["ppm"]
            x = d["zoom_peaks"]["data"]
            plt.plot(y, x, label="data")
            plt.plot(
                y[d["find_peaks"]["peaks"]],
                d["find_peaks"]["peak_data"]["peak_heights"],
                "o",
                label="peaks",
            )
            plt.plot(
                y[d["find_peaks"]["peak_data"]["peak_left_border"]],
                x[d["find_peaks"]["peak_data"]["peak_left_border"]],
                "*",
                label="left_ips",
            )
            plt.plot(
                y[d["find_peaks"]["peak_data"]["peak_right_border"]],
                x[d["find_peaks"]["peak_data"]["peak_right_border"]],
                "+",
                label="right_ips",
            )
            plt.title("find_peaks")
            plt.legend()
            plt.show()
            plt.close()

    def test_08_integrate_peaks(self):
        d = self._integrate_peaks()
        #       print(d["integrate_peaks"]["peak_data"]['integrals'])
        #        print(d["integrate_peaks"]["peak_data"]['from_cumintegrals'])
        if PLOT:
            x = d["zoom_peaks"]["ppm"]
            y = d["zoom_peaks"]["data"]
            plt.plot(x, y, label="data")
            # plt.plot(y[d["integrate_peaks"]["peaks"]],d["integrate_peaks"]["peak_data"]['peak_heights'],"o",label="peaks")
            cumi = d["integrate_peaks"]["peak_data"]["cum_integral"]
            cumi = cumi - cumi.min()
            cumi /= cumi.max() * 2
            cumi += 0.25
            plt.plot(x, cumi, "g--")
            incum = np.zeros(cumi.shape[0], dtype=bool)
            for i in range(d["integrate_peaks"]["peaks"].shape[0]):
                lb = d["integrate_peaks"]["peak_data"]["peak_left_border"][i]
                rb = d["integrate_peaks"]["peak_data"]["peak_right_border"][i]
                plt.fill_between(x=x[lb:rb], y1=y[lb:rb], alpha=0.5)
                incum[lb:rb] = True
            cumi[~incum] = np.nan
            plt.plot(x, cumi, "g", label="integral")

            plt.title("integrate_peaks")
            plt.legend()
            plt.show()
            plt.close()

    def test_09_ref_peaks(self):
        d = self._ref_peaks()
        ints = d["ref_peaks"]["integrals"]
        round_ints = np.round(ints).astype(int)
        int_diffs = round_ints - ints
        assert len(self.NORMED_PEAKS) == len(round_ints), [ints, self.NORMED_PEAKS]
        assert np.allclose(round_ints, self.NORMED_PEAKS), [
            round_ints,
            self.NORMED_PEAKS,
        ]
        assert np.allclose(int_diffs, self.PEAK_DIFF), [int_diffs, self.PEAK_DIFF]


class TestNMRMakgritek(AbstractNMRtester, unittest.TestCase):
    REF_PEAK = 3
    REF_PEAK_A = 1
    NORMED_PEAKS = [4, 1, 1, 4, 0]
    PEAK_DIFF = [0.2562199, -0.13086617, 0.0, -0.24022015, -0.34425704]
    TESTFILE = os.path.join(os.path.dirname(__file__), "testdata", "nmr", "magritek_1")


class TestNMRBruker(AbstractNMRtester, unittest.TestCase):
    REF_PEAK = 3.6
    REF_PEAK_A = 1
    NORMED_PEAKS = [1, 1, 1, 1, 1]
    PEAK_DIFF = [-0.46440374, -0.15092486, 0.25956847, 0.0, 0.23210654]
    TESTFILE = os.path.join(
        os.path.dirname(__file__), "testdata", "nmr", "bruker_1", "10"
    )
