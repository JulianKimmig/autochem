import json

#FOLDER = "C:\\Users\\be34gof\\Downloads\\TSU-141-E"
FOLDER = "C:\\Users\\be34gof\\Downloads\\25_11_2021"
MIN_PEAK_HEIGHT = 0.05  # minimum peak height relative to the largest peak
PEAK_BOARDER_RELATIVE_HEIGHT = 0.01  # peak height relative to the peak maximum which sets the integration limits
MAX_PEAK_WIDTH = 1  # maximum peak with, to limit very small and broad peaks
MIN_PEAK_DISTANCE = 0.1  # minimum peak distance, clsoer peaks are counted as one

ALLOW_PPM_STRETCH = False  # weather the ppm scale can be stretched to get expected peaks

FIXED_SCALE = [0, 6]
EXPECTED_PEAKS = [2.7, 3.9, 4.5]  # list of expected peaks for shift correction
MANUAL_PEAKS = [2.7, 3.9, 4.12, 4.3, 4.5]  # list of expected peaks for shift correction
MANUAL_PEAK_RANGES = [
    [2.40, 3.3],
    [3.4, 4.0],
    [4.0, 4.15],
    [4.15, 4.36],
    [4.37, 4.78],
]

SPECIES_PEAKS = [[3.9], [4.12], [4.3], [4.5],[2.7]]
SPECIES_PEAK_AREAS = [[2], [4], [6], [6],[6]]
SPECIES_PEAKS_NAMES = ["OME-1","OME-2","OME-3+","Trioxane","OME-CH3"]

# REFERENCE_PEAK=None
REFERENCE_PEAK = 2.7  # reference peak used as integration standart
REFERENCE_PEAK_AREA = 6  # area of the reference peak
REFERENCE_PEAK_WINDOW = 0.4  # maximum derivation from the reference peak

PLOT_INTERMEDIATES = False  # plot all intermediate steps
PLOT_RESULT = True  # plot result
SHOW_PLOTS = False  # live show plots, normally False

CREATE_TABLE = True  # results are stored as table files
RESULT_TABLE = True  # merge all results to on table
RECREATE = False  # recalc spec even if it is already in the results table
TABLE_TYPE = "xlsx"  # type of table data, use 'xlsx' for an excel-file or 'csv' for a csv-file



## dont do stuff here

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


from autochem.spectra.nmr.reader import read_nmr, NMRReadError
from autochem.utils.corrections import norm_data, shift_data, scale_data
from autochem.utils.signals.peak_detection import find_peaks, get_reference_peak, PeakNotFoundError, peak_integration, \
    merge_peaks_data, cut_peaks_data, factorize_peak_data, manual_peak_finder
from autochem.spectra.nmr.utils import zoom
from autochem.utils.corrections.shift import get_signal_shift

logger = logging.getLogger("autochem")
logger.setLevel("DEBUG")
MANUAL_PEAKS  = np.array(MANUAL_PEAKS)
EXPECTED_PEAKS = np.array(EXPECTED_PEAKS)
assert REFERENCE_PEAK in EXPECTED_PEAKS
assert len(MANUAL_PEAK_RANGES) == len(MANUAL_PEAKS)
assert len(SPECIES_PEAKS_NAMES) == len(SPECIES_PEAKS)

SPECIES_PEAKS = np.array(SPECIES_PEAKS)
SPECIES_PEAK_AREAS = np.array(SPECIES_PEAK_AREAS)
assert SPECIES_PEAK_AREAS.shape == SPECIES_PEAKS.shape

def find_nmrs(root, path_only=False, skip_path=None):
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


def plot_nmr(data, ppm, path=None, label=None, show=False,xlim=None):
    plt.plot(ppm, data, label=label)
    plt.xlabel("$\delta$ [ppm]")
    plt.legend(prop={'size': 6})
    if xlim:
        plt.xlim(*xlim)
    if path:
        plt.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close()


def work_spec(data, data_dict, path):
    processing=[]
    ppm_scale = data_dict["ppm_scale"]

    # plot raw data if wanted
    image_number = 0
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_raw_data.png"), label="raw_data",
                 show=SHOW_PLOTS)

    # First norm data between 0 and 1
    data, normata = norm_data(data)
    processing.append({
        "type":"linear norm",
        "parameter":normata,
        "comment":"norm between 0 and 1"}
    )
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_norm1.png"), label="norm1",
                 show=SHOW_PLOTS)

    data, normata = shift_data(data,np.median(data))
    processing.append({
        "type":"linear norm",
        "parameter":normata,
        "comment":"median to baseline"}
    )
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_norm2.png"), label="norm2",
                 show=SHOW_PLOTS)

    data, normata = scale_data(data,1/data.max())
    processing.append({
        "type":"linear norm",
        "parameter":normata,
        "comment":"maximum back to 1"}
    )
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_norm3.png"), label="norm3",
                 show=SHOW_PLOTS)


    # initial peak finder
    peaks, peak_data = manual_peak_finder(y=data, x=ppm_scale, peak_ranges=MANUAL_PEAK_RANGES)
    peaks, peak_data = find_peaks(y=data, x=ppm_scale,min_peak_height=MIN_PEAK_HEIGHT,
                                  rel_height=PEAK_BOARDER_RELATIVE_HEIGHT,
                                  min_distance=MIN_PEAK_DISTANCE,
                                  max_width=MAX_PEAK_WIDTH,
                                  rel_prominence=0.1,
                                  )

    # shift ppm scale to match expectations
    _m, _c = get_signal_shift(ppm_scale[peaks], EXPECTED_PEAKS, allow_stretch=ALLOW_PPM_STRETCH)
    original_ppm_scale=ppm_scale
    ppm_scale = ppm_scale * _m + _c
    processing.append({
        "type":"ppm shift",
        "parameter":[_m,_c],
        "comment":"shift ppm scale to best match expected peaks"}
    )
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_shifted_data.png"),
                 label="shifted_data", show=SHOW_PLOTS)



    peaks, peak_data = find_peaks(y=data, x=ppm_scale,min_peak_height=MIN_PEAK_HEIGHT,
                                  rel_height=PEAK_BOARDER_RELATIVE_HEIGHT,
                                  min_distance=MIN_PEAK_DISTANCE,
                                  max_width=MAX_PEAK_WIDTH,
                                  rel_prominence=0.5,
                                  )
    if PLOT_INTERMEDIATES:
        image_number += 1
        plt.plot(ppm_scale,data)
        for i in range(peaks.shape[0]):
            lb = peak_data['peak_left_border'][i]
            rb = peak_data['peak_right_border'][i]
            plt.fill_between(x=ppm_scale[lb:rb], y1=data[lb:rb], alpha=0.5, label=f"{ppm_scale[peaks[i]]:.2f} ppm")
        plt.xlim(*FIXED_SCALE)
        plt.legend()
        plt.savefig(os.path.join(path, f"img_{image_number}_first_peaks.png"), dpi=300)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    detected_peak_ranges=[]
    for mpr in MANUAL_PEAK_RANGES:
        peaks_in_range=(ppm_scale[peaks]>=mpr[0]) & (ppm_scale[peaks]<=mpr[1])
        if peaks_in_range.sum()==0:
            #detected_peak_ranges.append(None)
            continue
        if peaks_in_range.sum()>1:
            #filer by maximum size
            peaks_in_range[peaks_in_range] =  peak_data["peak_heights"][peaks_in_range] == peak_data["peak_heights"][peaks_in_range].max()

        if peaks_in_range.sum()>1:
            #filer by order
            fist=peaks_in_range.argmax()
            peaks_in_range[:] = False
            peaks_in_range[fist] = True

        detected_peak_ranges.append([
          #  max(
          #      mpr[0],
                ppm_scale[peak_data["peak_left_border"][peaks_in_range].min()],
          #  ),
        #    min(
         #       mpr[1],
                ppm_scale[peak_data["peak_right_border"][peaks_in_range].max()],
        #    ),
        ])

    detected_peak_ranges=np.array(detected_peak_ranges)
    peaks, peak_data = manual_peak_finder(y=data, x=ppm_scale,
                                          peak_ranges=[pr for pr in detected_peak_ranges if pr is not None],
                                          rel_height=PEAK_BOARDER_RELATIVE_HEIGHT,
                                          )


    # integrate_peaks
    peaks, peak_data = peak_integration(x=ppm_scale,
                                        y=data,
                                        peaks=peaks,
                                        peak_data=peak_data,
                                        )

    # integrate reference peak
    pidx, peak = get_reference_peak(
        ppm_scale[peaks],
        REFERENCE_PEAK,
        max_diff=REFERENCE_PEAK_WINDOW)

    scaler = REFERENCE_PEAK_AREA / peak_data["integrals"][pidx]
    peak_data = factorize_peak_data(peak_data, scale_factor=scaler)
    data, normata = scale_data(data,scaler)
    processing.append({
        "type":"linear norm",
        "parameter":normata,
        "comment":"scale for integral to match ref peak"}
    )
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_ref_scaled.png"),
                 label="ref scaled", show=SHOW_PLOTS,xlim=FIXED_SCALE)


    # zoom to relevant areas
    ppm_min = ppm_scale[peak_data["peak_left_border"].min()]
    ppm_max = ppm_scale[peak_data["peak_right_border"].max()]
    ppm_min, ppm_max = ppm_min - 0.1 * (ppm_max - ppm_min), ppm_max + 0.1 * (ppm_max - ppm_min)

    if FIXED_SCALE is not None:
        ppm_min, ppm_max = FIXED_SCALE

    data, ppm_scale, zoom_indices = zoom(
        data,
        ppm_scale,
        ppm_min,
        ppm_max,
    )

    processing.append({
        "type":"zoom",
        "parameter":zoom_indices,
        "comment":"to cut of unwanted/empty regions"}
    )
    if PLOT_INTERMEDIATES:
        image_number += 1
        plot_nmr(data, ppm_scale, path=os.path.join(path, f"img_{image_number}_zoomed.png"),
                 label="zoomed", show=SHOW_PLOTS)

    peaks, peak_data = cut_peaks_data(peaks, peak_data, *zoom_indices)
    # recreate integration data
    peaks, peak_data = peak_integration(x=ppm_scale,
                                        y=data,
                                        peaks=peaks,
                                        peak_data=peak_data,
                                        )


    # finishing up
    if PLOT_RESULT:
        image_number += 1

        plt.plot(ppm_scale, data, linewidth=1)


        for i in range(len(SPECIES_PEAKS_NAMES)):
            target_peaks=SPECIES_PEAKS[i]
            found=True
            for tp in target_peaks:
                if detected_peak_ranges.size==0 or ((detected_peak_ranges[:,0]<=tp) & (detected_peak_ranges[:,1]>=tp)).sum()==0:
                    #tp not found
                    found=False
                    break
            if not found:
                continue


            peak_positions=[]
            for p in target_peaks:
                peak_positions.append(np.abs(ppm_scale[peaks]-p).argmin())

            mean_height=peak_data["peak_heights"][peak_positions].mean()
            center_y=max(mean_height/2,peak_data["peak_heights"].max()/2)

            center_x=np.mean(ppm_scale[peaks][peak_positions])

            plt.text(center_x,center_y,SPECIES_PEAKS_NAMES[i],rotation="vertical",ha="center",alpha=0.6)
            plt.plot(
                [center_x]*len(target_peaks)+list(ppm_scale[peaks][peak_positions]),
                [center_y]*len(target_peaks)+list(peak_data["peak_heights"][peak_positions]/2),
                "k",alpha=0.3,linewidth=1)

        plt.plot(ppm_scale[peak_data["peak_median"]], peak_data["peak_heights"], "+", label="peaks median")
        plt.plot(ppm_scale[peak_data["peak_maximum"]], peak_data["peak_heights"], "+", label="peaks maximum")
        plt.plot(ppm_scale[peak_data["peak_mean"]], peak_data["peak_heights"], "+", label="peaks mean")

        cumi = peak_data["cum_integral"]
        cumi = cumi - cumi.min()
        cumi *= data.max() / (cumi.max() * 2)
        cumi += data.max() / 4
        plt.plot(ppm_scale, cumi, "g--")
        incum = np.zeros(cumi.shape[0], dtype=bool)
        for i in range(peaks.shape[0]):
            lb = peak_data['peak_left_border'][i]
            rb = peak_data['peak_right_border'][i]
            plt.fill_between(x=ppm_scale[lb:rb], y1=data[lb:rb], alpha=0.5, label=f"{ppm_scale[peaks[i]]:.2f} ppm")
            incum[lb:rb] = True
            cumi[~incum] = np.nan
        plt.plot(ppm_scale, cumi, "g", label="integral")

        plt.xlabel("$\delta$ [ppm]")
        #plt.yticks([])

        plt.legend(prop={'size': 6})
        plt.title(path)
        plt.savefig(os.path.join(path, f"img_{image_number}_peak_results.png"), dpi=300)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    df = pd.DataFrame({
        "ppm": ppm_scale[peaks],
        "ppm_max": ppm_scale[peak_data["peak_maximum"]],
        "ppm_mean": ppm_scale[peak_data["peak_mean"]],
        "left_border": ppm_scale[peak_data['peak_left_border']],
        "right_border": ppm_scale[peak_data['peak_right_border']],
        "area": peak_data["integrals"],
        "est nucl.": np.round(peak_data["integrals"]).astype(int),
    })

    try:
        df['startTime'] = dateutil.parser.parse(data_dict['acqu']['startTime'])
    except KeyError:
        df['startTime'] = dateutil.parser.parse("01.01.1990")

    df["Sample"] = data_dict['acqu'].get('Sample', "sample_name")

    with open(os.path.join(path, "processing_flow.txt"),"w+") as f:
        json.dump(processing,f,indent=4)
    if CREATE_TABLE:
        if TABLE_TYPE == "csv":
            df.to_csv(os.path.join(path, "signals_ac.csv"), index=False)
        elif TABLE_TYPE == "xlsx":
            df.to_excel(os.path.join(path, "signals_ac.xlsx"), merge_cells=True)
        else:
            raise ValueError(f"unknown TABLE_TYPE '{TABLE_TYPE}'")
    return df


def main():
    results_df = pd.DataFrame()
    if RESULT_TABLE:
        res_file = os.path.join(FOLDER, "results.xlsx")
        try:
            results_df = pd.read_excel(res_file, index_col=[0, 1, 2, 3])
        except FileNotFoundError:
            pass
    change = False

    def _sp(path):
        #if not os.path.basename(path).endswith("0"):
         #   return True
        return "path" in results_df.index.names and path in results_df.index.get_level_values(
            results_df.index.names.index('path')) and not RECREATE

    for data, data_dict in find_nmrs(FOLDER, skip_path=_sp):
        path = data_dict["path"]

        sdf = work_spec(data, data_dict, path)
        sdf["path"] = path
        sdf["peak"] = sdf.index.values + 1

        nindx = ["path", 'startTime', 'Sample', "peak"]
        sdf.set_index(nindx, inplace=True)
        results_df = pd.concat([results_df, sdf[~sdf.index.isin(results_df.index)]])
        results_df.update(sdf)
        change = True


    print(results_df)
    results_df.sort_values("startTime",inplace=True)

    if change and RESULT_TABLE:
        try:
            results_df.to_excel(res_file, merge_cells=True)
        except PermissionError:
            print(f"cannot write to file {res_file}, maybe it is opened in another program?")
    else:
        print("no changes detected")

    times = pd.to_datetime(results_df.index.get_level_values('startTime').unique()).values
    apd = np.zeros((len(times),SPECIES_PEAKS.size))*np.nan
    for i,t in enumerate(times):
        #_apd = []
        d = results_df.loc[results_df.index.get_level_values('startTime') == t]

        dist_matrix = np.abs( np.subtract.outer(SPECIES_PEAKS.flatten(),d["ppm"].values))

        while not np.isnan(dist_matrix).all():
            minidx = np.unravel_index(np.nanargmin(dist_matrix), dist_matrix.shape)
            apd[i,minidx[0]]=d["area"].values[minidx[1]]
            dist_matrix[minidx[0],:]=np.nan
            dist_matrix[:,minidx[1]]=np.nan


    tdiffa = times
    tdiffa = tdiffa - tdiffa.min()
    tdiffa = tdiffa / np.timedelta64(1, 's')
    for i in range(len(SPECIES_PEAKS.flatten())):
        plt.plot(tdiffa, apd[:, i], ".", label=f"{SPECIES_PEAKS.flatten()[i]} ppm")
        # opt_parms, parm_cov = curve_fit(_f,tdiff,apd[:,i])
        # plt.plot(tdiff,_f(tdiff,*opt_parms),label=str(EXPECTED_PEAKS[i]))
    plt.legend()
    plt.title("NMR signal area over time")
    plt.xlabel("t [s]")
    plt.ylabel("rel. area")
    plt.savefig(os.path.join(FOLDER, "nmr_area_over_time.png"),dpi=300)
    plt.show()
    plt.close()

    def _f(x, k, l, c):
        return k * np.exp(-l * x) + c



    #for i in range(len(SPECIES_PEAKS)):
    #    name=SPECIES_PEAKS_NAMES[i]
    #    peaks=SPECIES_PEAKS[i]
    #    pp=[]
    #    for p in peaks:
    #        pp.append(np.abs(MANUAL_PEAKS-p).argmin())


    #c_ome_eg=np.ones_like(apd[:,0])*apd[0,0]/SPECIES_PEAK_AREAS[0]
    #c_ome_eg=apd[:,0]/SPECIES_PEAK_AREAS[0]
    #c_ome_1=apd[:,1]/SPECIES_PEAK_AREAS[1]
    #c_ome_2=apd[:,2]/SPECIES_PEAK_AREAS[2]
    #c_ome_3=apd[:,3]/SPECIES_PEAK_AREAS[3]
    #c_triox=apd[:,4]/SPECIES_PEAK_AREAS[4]
    csa=apd/SPECIES_PEAK_AREAS.flatten()
    #f=c_ome_eg[0]/(c_ome_eg[0]+c_triox[0])
    #c_ome_eg*=f
    #c_ome_1*=f
    #c_ome_2*=f
    #c_ome_3*=f
    #c_triox*=f

    #for i in range(cs.shape[1]):
    #    plt.plot(tdiff,cs[:,i],".",label=SPECIES_PEAKS_NAMES[i])
    #plt.plot(tdiff,c_ome_1,".",label=SPECIES_PEAKS_NAMES[1])
    #plt.plot(tdiff,c_ome_2,".",label=SPECIES_PEAKS_NAMES[2])
    #plt.plot(tdiff,c_ome_3,".",label=SPECIES_PEAKS_NAMES[3])
    #plt.plot(tdiff,c_triox,".",label=SPECIES_PEAKS_NAMES[4])
    #plt.plot(tdiff,c_ome_1+c_ome_2+c_ome_3,".",label="sum OME")
    #plt.legend()
    #plt.savefig(os.path.join(FOLDER, "species_conc.png"),dpi=300)
    #plt.show()
    #plt.close()

    def ffill(arr):
        arr=arr.copy()
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        arr = arr[idx]
        return arr

    def moving_average(a, n=3) :
        n=min(n,a.shape[0])
        a=a.copy()
        ret = np.cumsum(np.nan_to_num(a), dtype=float,axis=0)
        ret[n:] = ret[n:] - ret[:-n]
        ret[n - 1:]/=n
        if n>1:
            ret[:n - 1]=(ret[:n - 1].T/np.arange(1,n)).T
        return ret

    os.makedirs(os.path.join(FOLDER,"timelaps",),exist_ok=True)
    for ti in range(2,csa.shape[0],1):
        cs=csa[:ti]
        tdiff=tdiffa[:ti]
        impath=os.path.join(FOLDER,"timelaps", f"{ti:06d}.png")
        if os.path.exists(impath):
            continue
        mvg_avg = moving_average(cs,30)

        mvg_avg_diff = np.abs(cs-mvg_avg)
        mvg_avg_diff_mvg_avg = moving_average(mvg_avg_diff,10)
        th=1
        diff_mvg_avg_diff_mvg_avg = np.abs(mvg_avg_diff_mvg_avg-mvg_avg_diff)/mvg_avg_diff_mvg_avg

        outlier=diff_mvg_avg_diff_mvg_avg>th

        n_cs=cs.copy()
        n_cs[outlier]=np.nan

        if n_cs.ndim>1:
            for i in range(n_cs.shape[1]):
                n_cs[:,i]=ffill(n_cs[:,i])
        else:
            n_cs=ffill(n_cs)

        n_mvg_avg = moving_average(n_cs,30)

        ep_th=0.003
        lp_th=0.0012
        grad = np.abs(np.gradient(n_mvg_avg,axis=0))
        avg_grad = moving_average(grad,30)

        entry_point = avg_grad.mean(1)
        ep_idx=(entry_point>ep_th).argmax()

        lp_idx=(entry_point[ep_idx:]>lp_th).argmin()+ep_idx


        f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]},figsize=(10,10))
        for i in range(n_cs.shape[1]):
            ax1.plot(tdiff,n_cs[:,i],".",label=SPECIES_PEAKS_NAMES[i],alpha=0.6)
        for i in range(n_cs.shape[1]):
            ax1.plot(tdiff,n_mvg_avg[:,i],label=SPECIES_PEAKS_NAMES[i]+" mean")

        ax2.plot(tdiff,entry_point,label="mean gradient")

        if ep_idx>0:
            ax2.vlines(tdiff[ep_idx],0,entry_point.max(),color="green",label="rxn start")


            if lp_idx>ep_idx:
                ax2.vlines(tdiff[lp_idx],0,entry_point.max(),color="red",label="rxn end")

        ax1.legend(loc='upper right')
        ax2.legend()
        plt.savefig(impath,dpi=200)
        #plt.show()
        plt.close()

    return
    images=[]
    import imageio
    for f in sorted(os.listdir(os.path.join(FOLDER,"timelaps"))):
        images.append(imageio.imread(os.path.join(FOLDER,"timelaps",f)))
    imageio.mimsave(os.path.join(FOLDER,"timelaps.gif"), images)

    #moving_average(np.arange(100).reshape(25,4),5)
    #popt,pcov = curve_fit(poly, tdiff, c_ome_eg,bounds=([0],[10]))

   #for i in range(cs.shape[1]):
   #     plt.plot(tdiff,cs[:,i],".",label=SPECIES_PEAKS_NAMES[i],alpha=0.6)
   # for i in range(cs.shape[1]):
   #     plt.plot(tdiff,mvg_avg[:,i],label=SPECIES_PEAKS_NAMES[i])
   # plt.legend()
   # plt.show()
   # plt.close()


    #for i in range(cs.shape[1]):
    #    plt.plot(tdiff,mvg_avg_diff[:,i],label=SPECIES_PEAKS_NAMES[i])
    #    plt.plot(tdiff,mvg_avg_diff_mvg_avg[:,i],label=SPECIES_PEAKS_NAMES[i])

    #plt.legend()
    #plt.show()
    #plt.close()


    #for i in range(cs.shape[1]):
    #    plt.plot(tdiff,diff_mvg_avg_diff_mvg_avg[:,i],label=SPECIES_PEAKS_NAMES[i])
    #plt.hlines(th,xmin=tdiff.min(),xmax=tdiff.max())

    #plt.legend()
    #plt.show()
    #plt.close()

    return
    for i in range(n_cs.shape[1]):
        plt.plot(tdiff,n_cs[:,i],".",label=SPECIES_PEAKS_NAMES[i],alpha=0.6)
    for i in range(n_cs.shape[1]):
        plt.plot(tdiff,n_mvg_avg[:,i],label=SPECIES_PEAKS_NAMES[i])
    plt.legend()
    plt.show()
    plt.close()


    for i in range(n_cs.shape[1]):
        #plt.plot(tdiff,grad[:,i],label=SPECIES_PEAKS_NAMES[i])
        plt.plot(tdiff,avg_grad[:,i],label=SPECIES_PEAKS_NAMES[i])

    plt.plot(tdiff,entry_point,label="mean")

    plt.vlines(tdiff[ep_idx],0,1)


    if lp_idx>0:
        plt.vlines(tdiff[lp_idx],0,1)

    plt.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
