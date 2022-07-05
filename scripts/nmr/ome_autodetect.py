import logging
import sys, os

import dateutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

if __name__ == '__main__':
    ddir = os.path.dirname(os.path.abspath(__file__))
    while "autochem" not in os.listdir(ddir):
        ddir = os.path.dirname(ddir)
    sys.path.insert(0, ddir)
    sys.path.append(ddir)

    logging.basicConfig()

FOLDER = "C:\\Users\\be34gof\Downloads\\20220211084759-PE-237-1-5°C"
MIN_PEAK_HEIGHT= 0.1 # minimum peak height relative to the largest peak
PEAK_BOARDER_RELATIVE_HEIGHT = 0.1 # peak height relative to the peak maximum which sets the integration limits
MAX_PEAK_WIDTH = 1  # maximum peak with, to limit very small and broad peaks
MIN_PEAK_DISTANCE = 0.1 # minimum peak distance, clsoer peaks are counted as one
EXPECTED_PEAKS = [0.8,2.8, 3.9,4.12,4.3, 4.5] # list of expected peaks for shift correction
ALLOW_PPM_STRETCH = False # weather the ppm scale can be stretched to get expected peaks

FIXED_SCALE=[0,6] # if
MANUAL_PEAK_RANGES=[
    [0.29,1.24],
    [2.5,3.3],
    [3.8,4.05],
    [4.05,4.23],
    [4.23,4.47],
    [4.47,4.78],
]


#REFERENCE_PEAK=None
REFERENCE_PEAK=0.8 # reference peak used as integration standart
REFERENCE_PEAK_AREA = 1 # area of the reference peak
REFERENCE_PEAK_WINDOW = 0.4 # maximum derivation from the reference peak

PLOT_INTERMEDIATES = False # plot all intermediate steps
PLOT_RESULT = True # plot result
SHOW_PLOTS = False # live show plots, normally False


CREATE_TABLE = True # results are stored as table files
RESULT_TABLE = True # merge all results to on table
RECREATE = False # recalc spec even if it is already in the results table
TABLE_TYPE = "xlsx" # type of table data, use 'xlsx' for an excel-file or 'csv' for a csv-file

from autochem.spectra.nmr.reader import read_nmr, NMRReadError
from autochem.utils.corrections import norm_data
from autochem.utils.signals.peak_detection import find_peaks, get_reference_peak, PeakNotFoundError, peak_integration, \
    merge_peaks_data, cut_peaks_data, factorize_peak_data, manual_peak_finder
from autochem.spectra.nmr.utils import zoom
from autochem.utils.corrections.shift import get_signal_shift
from autochem.utils.corrections.baseline import rubberband_correction, median_correction

logger = logging.getLogger("autochem")
logger.setLevel("DEBUG")

EXPECTED_PEAKS = np.array(EXPECTED_PEAKS)

def find_nmrs(root, path_only=False,skip_path=None):
    for path, folder, files in os.walk(root):
        if skip_path is not None:
            if skip_path(path):
                continue
        try:
            data, udict = read_nmr(path, preprocess=True)
            logger.info("read '{}' as '{}'".format(path, udict["datatype"]))
            if path_only:
                yield path
            else:
                udict["path"] = path
                yield (data, udict)
        except NMRReadError:
            pass


def plot_nmr(data, ppm, path=None,label=None,show=False):
    plt.plot(ppm, data,label=label)
    plt.xlabel("$\delta$ [ppm]")
    plt.legend(prop={'size': 6})
    if path:
        plt.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close()


def work_spec(data,data_dict,path):
    ppm_scale = data_dict["ppm_scale"]

    # plot raw data if wanted
    image_number = 0
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_raw_data.png"),label="raw_data",show=SHOW_PLOTS)


    # First norm data between 0 and 1
    data, normata = norm_data(data)
    data, bl_data_rb=median_correction(data)
    data/=data.max()

    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_normed_data.png"),label="normed_data",show=SHOW_PLOTS)

    # initial peak finder
    if MANUAL_PEAK_RANGES is not None:
        peaks, peak_data = manual_peak_finder(y=data, x=ppm_scale,peak_ranges=MANUAL_PEAK_RANGES)
    else:
        peaks, peak_data = find_peaks(y=data, x=ppm_scale, min_peak_height=MIN_PEAK_HEIGHT,
                                  rel_height=PEAK_BOARDER_RELATIVE_HEIGHT/10,
                                  max_width=MAX_PEAK_WIDTH,
                                  min_distance=MIN_PEAK_DISTANCE,
                                  center="median"
                                  )


    #shift ppm scale to match expectations
    if len(EXPECTED_PEAKS)>0:
        _m,_c = get_signal_shift(ppm_scale[peaks],EXPECTED_PEAKS,allow_stretch=ALLOW_PPM_STRETCH)
        ppm_scale=ppm_scale*_m + _c

        if PLOT_INTERMEDIATES:
            image_number += 1
            plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_shifted_data.png"),label="shifted_data",show=SHOW_PLOTS)


    #integrate_peaks
    peaks, peak_data = peak_integration(x=ppm_scale,
                                        y=data,
                                        peaks=peaks,
                                        peak_data=peak_data,
                                        )


    # integrate reference peak
    if REFERENCE_PEAK is not None:
        pidx=None
        try:
            pidx, peak = get_reference_peak(
                ppm_scale[peaks],
                REFERENCE_PEAK,
                max_diff=REFERENCE_PEAK_WINDOW)
        except PeakNotFoundError:
            in_ppm=(ppm_scale>=REFERENCE_PEAK-REFERENCE_PEAK_WINDOW) & (ppm_scale<=REFERENCE_PEAK+REFERENCE_PEAK_WINDOW)
            sub_peaks, sub_peak_data = find_peaks(y=data[in_ppm], x=ppm_scale[in_ppm],
                                                  rel_height=PEAK_BOARDER_RELATIVE_HEIGHT, rel_prominence=0,
                                                  min_distance=REFERENCE_PEAK_WINDOW*2,
                                                  max_width=MAX_PEAK_WIDTH,
                                                  center="median"
                                                  )

            sub_peaks, sub_peak_data = peak_integration(x=ppm_scale[in_ppm],
                                                        y=data[in_ppm],
                                                        peaks=sub_peaks,
                                                        peak_data=sub_peak_data,
                                                        )

            peaks, peak_data,_ppm = merge_peaks_data(peaks, sub_peaks, peaks_data1=peak_data, peaks_data2=sub_peak_data, x1=ppm_scale,
                                                     x2=ppm_scale[in_ppm], peaks2_y_shift=in_ppm.argmax())

            try:
                pidx, peak = get_reference_peak(
                    ppm_scale[peaks],
                    REFERENCE_PEAK,
                    max_diff=REFERENCE_PEAK_WINDOW)
            except PeakNotFoundError:
                logger.warning("no ref peak found")

        if pidx is not None:
            scaler=REFERENCE_PEAK_AREA / peak_data["integrals"][pidx]
            peak_data = factorize_peak_data(peak_data,scale_factor = scaler)
            data*=scaler



    #zoom to relevant areas
    ppm_min = ppm_scale[peak_data["peak_left_border"].min()]
    ppm_max = ppm_scale[peak_data["peak_right_border"].max()]
    ppm_min,ppm_max=ppm_min-0.1*(ppm_max-ppm_min),ppm_max+0.1*(ppm_max-ppm_min)

    if FIXED_SCALE is not None:
        ppm_min, ppm_max = FIXED_SCALE

    data, ppm_scale,zoom_indices = zoom(
        data,
        ppm_scale,
        ppm_min,
        ppm_max,
    )

    peaks, peak_data = cut_peaks_data(peaks, peak_data,*zoom_indices)


    #recreate integration data
    peaks, peak_data = peak_integration(x=ppm_scale,
                                        y=data,
                                        peaks=peaks,
                                        peak_data=peak_data,
                                        )

    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_zoomed_data.png"),label="zoomed_data",show=SHOW_PLOTS)



    #finishing up
    if PLOT_RESULT:
        image_number += 1

        plt.plot(ppm_scale,data,linewidth=1)
        plt.plot(ppm_scale[peaks], peak_data["peak_heights"], "+", label="peaks median")
        plt.plot(ppm_scale[peak_data["peak_maximum"]], peak_data["peak_heights"], "+", label="peaks maximum")
        plt.plot(ppm_scale[peak_data["peak_mean"]], peak_data["peak_heights"], "+", label="peaks mean")

        cumi = peak_data["cum_integral"]
        cumi = cumi - cumi.min()
        cumi *= data.max()/(cumi.max()*2)
        cumi += data.max()/4
        plt.plot(ppm_scale, cumi, "g--")
        incum = np.zeros(cumi.shape[0], dtype=bool)
        for i in range(peaks.shape[0]):
            lb = peak_data['peak_left_border'][i]
            rb = peak_data['peak_right_border'][i]
            plt.fill_between(x=ppm_scale[lb:rb], y1=data[lb:rb], alpha=0.5,label=f"peak {i+1}")
            incum[lb:rb] = True
            cumi[~incum] = np.nan
        plt.plot(ppm_scale, cumi, "g", label="integral")

        plt.xlabel("$\delta$ [ppm]")
        plt.yticks([])

        plt.legend(prop={'size': 6})
        plt.title(path)
        plt.savefig(os.path.join(path, f"img_{image_number}_peak_results.png"),dpi=300)
        if SHOW_PLOTS:
            plt.show()
        plt.close()


    df = pd.DataFrame( {
        "ppm":ppm_scale[peaks],
        "ppm_max":ppm_scale[peak_data["peak_maximum"]],
        "ppm_mean":ppm_scale[peak_data["peak_mean"]],
        "left_border":ppm_scale[peak_data['peak_left_border']],
        "right_border":ppm_scale[peak_data['peak_right_border']],
        "area":peak_data["integrals"],
        "est nucl.":np.round(peak_data["integrals"]).astype(int)
    })

    try:
        df['startTime'] = dateutil.parser.parse(data_dict['acqu']['startTime'])
    except KeyError:
        df['startTime'] = dateutil.parser.parse("01.01.1990")

    df["Sample"] = data_dict['acqu'].get('Sample',"sample_name")

    if CREATE_TABLE:
        if TABLE_TYPE == "csv":
            df.to_csv(os.path.join(path,"signals_ac.csv"),index=False)
        elif TABLE_TYPE == "xlsx":
            df.to_excel(os.path.join(path,"signals_ac.xlsx"),merge_cells=True)
        else:
            raise ValueError(f"unknown TABLE_TYPE '{TABLE_TYPE}'")
    return df

def main():
    results_df=pd.DataFrame()
    results_df.index.get_level_values
    if RESULT_TABLE:
        res_file=os.path.join(FOLDER,"results.xlsx")
        try:
            results_df=pd.read_excel(res_file,index_col=[0,1,2,3])
        except FileNotFoundError:
            pass
    change=False

    def _sp(path):
        return "path" in results_df.index.names and path in results_df.index.get_level_values(results_df.index.names.index('path')) and not RECREATE

    for data, data_dict in find_nmrs(FOLDER,skip_path=_sp):
        path = data_dict["path"]
        if "path" in results_df.index.names and path in results_df.index.get_level_values(results_df.index.names.index('path')) and not RECREATE:
            continue
        sdf = work_spec(data, data_dict,path)
        sdf["path"]=path
        sdf["peak"]=sdf.index.values + 1

        nindx = ["path",'startTime','Sample',"peak"]
        sdf.set_index(nindx, inplace = True)
        results_df = pd.concat([results_df, sdf[~sdf.index.isin(results_df.index)]])
        results_df.update(sdf)
        change=True


    if change and RESULT_TABLE:
        try:
            results_df.to_excel(res_file,merge_cells=True)
        except PermissionError:
            print(f"cannot write to file {res_file}, maybe it is opened in another program?")
    else:
        print("no changes detected")

    times=pd.to_datetime(results_df.index.get_level_values('startTime').unique()).values
    apd=[]
    for t in times:
        _apd=[]
        d = results_df.loc[results_df.index.get_level_values('startTime') == t]
        for ex in EXPECTED_PEAKS:
            _apd.append(
                d["area"].values[np.abs(d["ppm"]-ex).argmin()]
            )
        apd.append(_apd)
       # print(_apd)
       # break

    apd=np.array(apd)
    def _f(x,k,l,c):
        return k*np.exp(-l*x)+c
    tdiff=times
    tdiff=tdiff-tdiff.min()
    tdiff= tdiff / np.timedelta64(1,'s')
    for i in range(len(EXPECTED_PEAKS)):
        plt.plot(tdiff,apd[:,i],".",label=str(EXPECTED_PEAKS[i]))
        #opt_parms, parm_cov = curve_fit(_f,tdiff,apd[:,i])
        #plt.plot(tdiff,_f(tdiff,*opt_parms),label=str(EXPECTED_PEAKS[i]))
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("rel. area")
    plt.savefig(os.path.join(FOLDER,"areas_over_time.png"),dpi=300)
    plt.show()







if __name__ == '__main__':
    main()
