import os
import time

import dateutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from autochem.spectra.nmr.reader import NMRReadError, read_nmr
from autochem.spectra.nmr.utils import zoom
from autochem.utils.corrections import norm_data
from autochem.utils.corrections.baseline import rubberband_correction, asymmetric_least_squares_smoothing
from autochem.utils.signals.peak_detection import find_peaks, peak_integration, get_reference_peak, PeakNotFoundError, \
    merge_peaks_data


def work_path(path, min_peak_height=0.02, min_peak_distance=0.1, max_peak_width=2, ref_peak=None,ref_peak_window=0.3, ref_area=1,
              peak_borders_rel_height=0.01,
              create_image=False, create_csv=False):
    data, udict = read_nmr(path, preprocess=True)
    print("read '{}' as {}".format(path,udict["datatype"]))
    ppm=udict["ppm_scale"]

    if create_image:
        plt.plot(ppm,data,linewidth=1,label="raw_data")
        plt.xlabel("$\delta$ [ppm]")
        plt.legend(prop={'size': 6})
        plt.savefig(os.path.join(path,"img_raw_data.png"),dpi=300)
        plt.close()

    data, normed = norm_data(data)







    peaks, peak_data = find_peaks(y=data, x=ppm, min_peak_height=min_peak_height, rel_height=peak_borders_rel_height/10,
                                  max_width=max_peak_width
                                  )

    ppm_min = ppm[peak_data["peak_left_border"].min()]
    ppm_max = ppm[peak_data["peak_right_border"].max()]

    if ref_peak is not None:
        ppm_min=min(ppm_min,ref_peak-ref_peak_window)
        ppm_max=max(ppm_max,ref_peak+ref_peak_window)

    ppm_min,ppm_max=ppm_min-0.1*(ppm_max-ppm_min),ppm_max+0.1*(ppm_max-ppm_min)

    data, ppm = zoom(
        data,
        ppm,
        ppm_min,
        ppm_max,
    )

    if create_image:
        plt.plot(ppm,data,linewidth=1,label="zoomed_data")
        plt.xlabel("$\delta$ [ppm]")
        plt.legend(prop={'size': 6})
        plt.savefig(os.path.join(path,"img_zoomed_data.png"),dpi=300)
        plt.close()


    if create_image:
        plt.plot(ppm,data,linewidth=1,label="data")



    data, bl_data_als= asymmetric_least_squares_smoothing(data,lam=10**5,p=0.0001)
    if create_image:
        plt.plot(ppm,data,linewidth=1,label="baseline substracted als")

    data, bl_data_rb=rubberband_correction(ppm,data)
    if create_image:
        plt.plot(ppm,data,linewidth=1,label="baseline substracted rubberband")

    if create_image:
        plt.plot(ppm,bl_data_rb["baseline"],linewidth=1,label="baseline rubberband")
        plt.plot(ppm,bl_data_als["baseline"],linewidth=1,label="baseline als")
        plt.xlabel("$\delta$ [ppm]")
        plt.legend(prop={'size': 6})
        plt.savefig(os.path.join(path,"img_bl_subs.png"),dpi=300)
        plt.close()

    data, normed = norm_data(data)


    peaks, peak_data = find_peaks(y=data, x=ppm, min_peak_height=min_peak_height,
                                  rel_height=peak_borders_rel_height, rel_prominence=0.4,min_distance=min_peak_distance,
                                  max_width=max_peak_width,
                                  center="median"
                                  )

    peaks, peak_data = peak_integration(x=ppm,
                                        y=data,
                                        peaks=peaks,
                                        peak_data=peak_data,
                                        )

    if ref_peak is not None:
        try:
            pidx, peak = get_reference_peak(
                ppm[peaks],
                ref_peak,
                max_diff=ref_peak_window)
        except PeakNotFoundError:
            in_ppm=(ppm>=ref_peak-ref_peak_window) & (ppm<=ref_peak+ref_peak_window)
            sub_peaks, sub_peak_data = find_peaks(y=data[in_ppm]/data[in_ppm].max(), x=ppm[in_ppm], min_peak_height=1,
                                          rel_height=peak_borders_rel_height, rel_prominence=0,min_distance=ref_peak_window,
                                          max_width=max_peak_width,
                                          center="median"
                                          )

            sub_peaks, sub_peak_data = peak_integration(x=ppm[in_ppm],
                                                y=data[in_ppm]/data[in_ppm].max(),
                                                peaks=sub_peaks,
                                                peak_data=sub_peak_data,
                                                )

            peaks, peak_data,_ppm = merge_peaks_data(peaks, sub_peaks, peaks_data1=peak_data, peaks_data2=sub_peak_data, x1=ppm,
                                                x2=ppm[in_ppm], peaks2_y_shift=in_ppm.argmax(), peaks2_norm_fac=data[in_ppm].max())

            pidx, peak = get_reference_peak(
                ppm[peaks],
                ref_peak,
                max_diff=ref_peak_window)

        normf = ref_area / peak_data["integrals"][pidx]
        peak_data["integrals"]*= normf
        peak_data["cum_integral"]*= normf,

    if create_image:
        plt.plot(ppm,data,linewidth=1)
        plt.plot(ppm[peaks], peak_data["peak_heights"], "+", label="peaks median")
        plt.plot(ppm[peak_data["peak_maximum"]], peak_data["peak_heights"], "+", label="peaks maximum")
        plt.plot(ppm[peak_data["peak_mean"]], peak_data["peak_heights"], "+", label="peaks mean")

        cumi = peak_data["cum_integral"]
        cumi = cumi - cumi.min()
        cumi /= cumi.max() * 2
        cumi += 0.25
        plt.plot(ppm, cumi, "g--")
        incum = np.zeros(cumi.shape[0], dtype=bool)
        for i in range(peaks.shape[0]):
            lb = peak_data['peak_left_border'][i]
            rb = peak_data['peak_right_border'][i]
            plt.fill_between(x=ppm[lb:rb], y1=data[lb:rb], alpha=0.5,label=f"peak {i+1}")
            incum[lb:rb] = True
            cumi[~incum] = np.nan
        plt.plot(ppm, cumi, "g", label="integral")

        plt.xlabel("$\delta$ [ppm]")
        plt.yticks([])

        plt.legend(prop={'size': 6})

        plt.savefig(os.path.join(path,"img_signals_ac.png"),dpi=300)
        plt.close()

    df = pd.DataFrame( {
        "ppm":ppm[peaks],
        "ppm_max":ppm[peak_data["peak_maximum"]],
        "ppm_mean":ppm[peak_data["peak_mean"]],
        "area":peak_data["integrals"],
        "est nucl.":np.round(peak_data["integrals"]).astype(int)
    })

    try:
        df['startTime'] = dateutil.parser.parse(udict['acqu']['startTime'])
    except KeyError:
        df['startTime'] = dateutil.parser.parse("01.01.1990")

    df["Sample"] = udict['acqu'].get('Sample',"sample_name")

    if create_csv:
        df.to_csv(os.path.join(path,"signals_ac.csv"),index=False)

    return df

def main(parser):
    parser.add_argument("--min_peak_height", type=float,help='minimum peak height, relative to maximum',default=0.02)
    parser.add_argument("--min_peak_distance", type=float,help='minimum peak distance',default=0.1)
    parser.add_argument("--max_peak_width", type=float,help='minimum peak width',default=2)
    parser.add_argument("--ref_peak", type=float,help='reference peak position',default=None)
    parser.add_argument("--ref_peak_window", type=float,help='max difference to given ref peak',default=0.3)
    parser.add_argument("--ref_area", type=float,help='reference peak area',default=1)
    parser.add_argument("--peak_borders_rel_height", type=float,help='rel peak height indicating peak boarder',default=0.01)
    parser.add_argument("--csv", help='create result csv', default=False, action='store_true')
    parser.add_argument("--img", help='create result image', default=False, action='store_true')

    parser.add_argument("--results",help='create results file in source folder', default=False, action='store_true')
    parser.add_argument("--recreate",help='if the sample is already in the results file it wont be parsed if this flag is not set', default=False, action='store_true')

    parser.add_argument("--continuous",help='runs the program in a loop for continuous observation', default=False, action='store_true')

    args = parser.parse_args()


    def run_once():
        results_df=pd.DataFrame()
        args.folder=os.path.abspath(args.folder)
        if args.results:
            res_file=os.path.join(args.folder,"results.xlsx")
            try:
                results_df=pd.read_excel(res_file,index_col=[0,1,2,3])
            except FileNotFoundError:
                pass
        change=False
        for path,folder,files in os.walk(args.folder):
            try:
                if "path" in results_df.index.names and path in results_df.index.get_level_values(results_df.index.names.index('path')) and not args.recreate:
                    continue
                sdf = work_path(path, min_peak_height=args.min_peak_height, min_peak_distance=args.min_peak_distance,
                          max_peak_width=args.max_peak_width,ref_peak=args.ref_peak,ref_area=args.ref_area,ref_peak_window=args.ref_peak_window,
                          create_csv=args.csv,
                          create_image=args.img)
                sdf["path"]=path
                sdf["peak"]=sdf.index.values + 1

                nindx = ["path",'startTime','Sample',"peak"]
                sdf.set_index(nindx, inplace = True)
                results_df = pd.concat([results_df, sdf[~sdf.index.isin(results_df.index)]])
                results_df.update(sdf)
                change=True
            except NMRReadError:
                pass

        if change and args.results:
            #writer = pd.ExcelWriter(, engine='xlsxwriter')
            try:
                results_df.to_excel(res_file,merge_cells=True)
            except PermissionError:
               print(f"cannot write to file {res_file}, maybe it is opened in another program?")
        else:
            print("no changes detected")

    while args.continuous:
        run_once()
        args.recreate=False # if true only once!
        time.sleep(10)
    run_once()