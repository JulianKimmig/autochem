import json

#FOLDER = "C:\\Users\\be34gof\\Downloads\\TSU-141-E"
from pprint import pprint

FOLDER = [
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220202115053-PE-236-1",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220204071912-PE-236-2",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220207081529-PE-236-3",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220209093938-PE-236-4",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220218100335_PE-236-5",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220216095015_PE-238-1",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220217124558_PE-239-1_30°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220217143734_PE-239-2_25°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220217164550_PE-239-3_20°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220221090332 PE-239-4_15°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220222092051 PE-239-5_10°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220211084759-PE-237-1-5°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220214034707_PE-237_3-15°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220214102859_PE-237-4-10°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220214161917_PE-237-5-25°C",
    "C:\\Users\\be34gof\\Downloads\\autonmr\\20220215094536_PE-237-6-30°C",
]

MIN_PEAK_HEIGHT = 0.05  # minimum peak height relative to the largest peak
PEAK_BOARDER_RELATIVE_HEIGHT = 0.01  # peak height relative to the peak maximum which sets the integration limits
MAX_PEAK_WIDTH = 1  # maximum peak with, to limit very small and broad peaks
MIN_PEAK_DISTANCE = 0.1  # minimum peak distance, clsoer peaks are counted as one

ALLOW_PPM_STRETCH = False  # weather the ppm scale can be stretched to get expected peaks

FIXED_SCALE = [0, 6]
EXPECTED_PEAKS = [2.7, 3.9, 4.5]  # list of expected peaks for shift correction
MANUAL_PEAKS = [2.7, 3.9, 4.12, 4.2, 4.5]  # list of expected peaks for shift correction
MANUAL_PEAK_RANGES = [
    [2.40, 3.],
    [3.4, 4.0],
    [4.0, 4.17],
    [4.1701, 4.3499],
    [4.35, 4.78],
]

SPECIES_PEAKS = [[3.9], [4.12], [4.2], [4.5],[2.7]]
SPECIES_PEAK_AREAS = [[2], [4], [6], [6],[6]]
SPECIES_MW = [76.1 , 76.1+30.03 , 76.1+3*30.03 , 90.08 , 0]
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
    #peaks, peak_data = manual_peak_finder(y=data, x=ppm_scale, peak_ranges=MANUAL_PEAK_RANGES)
    peaks, peak_data = find_peaks(y=data, x=ppm_scale,min_peak_height=MIN_PEAK_HEIGHT,
                                  rel_height=PEAK_BOARDER_RELATIVE_HEIGHT,
                                  min_distance=MIN_PEAK_DISTANCE,
                                  max_width=MAX_PEAK_WIDTH,
                                  rel_prominence=0.1
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
                                  peak_ranges=MANUAL_PEAK_RANGES
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

    peak_targets=ppm_scale[peaks].copy()
    flatted_specie_peaks=[]
    flatted_specie_peaks_indices=[]
    flatted_specie_peaks_areas=[]
    for i,pl in enumerate(SPECIES_PEAKS):
        flatted_specie_peaks.extend(pl)
        flatted_specie_peaks_areas.extend(SPECIES_PEAK_AREAS[i])
        flatted_specie_peaks_indices.extend([i]*len(pl))
    flatted_specie_peaks=np.array(flatted_specie_peaks)
    flatted_specie_peaks_areas=np.array(flatted_specie_peaks_areas)
    flatted_specie_peaks_indices=np.array(flatted_specie_peaks_indices)

    diff_matrix = np.abs(np.subtract.outer(flatted_specie_peaks,peak_targets))
    sorted_indices=[]
    while np.any(~np.isnan(diff_matrix)):
        induices =np.unravel_index(np.nanargmin(diff_matrix), diff_matrix.shape)
        sorted_indices.append(induices)
        diff_matrix[induices]=np.nan
    sorted_indices=np.array(sorted_indices).astype(float)

    targets={}
    #print(flatted_specie_peaks,peak_targets)
    #print(sorted_indices)
    while sorted_indices.shape[0]>0:
        sorted_indices=sorted_indices[~np.isnan(sorted_indices)].reshape(-1,2)
        found=False

        for i,(ix1,ix2) in enumerate(sorted_indices):
            ix1=int(ix1)
            ix2=int(ix2)
            species_peak=flatted_specie_peaks[ix1]
            target_peak=peak_targets[ix2]
            dec_range=detected_peak_ranges[ix2]
            if dec_range[0]<=species_peak and dec_range[1]>=species_peak:
                found=True
                targets[target_peak]=ix1
                sorted_indices[sorted_indices[:,0]==ix1]=np.nan
                sorted_indices[sorted_indices[:,1]==ix2]=np.nan
                break

        if not found:
            sorted_indices[:]=np.nan

        if sorted_indices.shape[0]==0:
            break

    #print(targets)
    df = pd.DataFrame({
        "ppm": ppm_scale[peaks],
        "ppm_max": ppm_scale[peak_data["peak_maximum"]],
        "species": [(SPECIES_PEAKS_NAMES[flatted_specie_peaks_indices[targets.get(p)]] if targets.get(p) is not None else "") for p in peak_targets],
        "species_peak": [(flatted_specie_peaks[targets.get(p)] if targets.get(p) is not None else np.nan) for p in peak_targets],
        "species_peak_area":[(flatted_specie_peaks_areas[targets.get(p)] if targets.get(p) is not None else np.nan) for p in peak_targets],
        "ppm_mean": ppm_scale[peak_data["peak_mean"]],
        "ppm_median": ppm_scale[peak_data["peak_median"]],
        "left_border": ppm_scale[peak_data['peak_left_border']],
        "height": peak_data["peak_heights"],
        "right_border": ppm_scale[peak_data['peak_right_border']],
        "area": peak_data["integrals"],
        "est nucl.": np.round(peak_data["integrals"]).astype(int),
    })
    df["estimated_rel_conc"]=df["area"]/df["species_peak_area"]
    df["Sample"] = data_dict['acqu'].get('Sample', "sample_name")

    df.sort_values("ppm",inplace=True)
    if PLOT_RESULT:
        image_number += 1

        plt.plot(ppm_scale, data, linewidth=1)

        max_height=df["height"].max()

        for r,d in df.iterrows():
            center_x=d["ppm_mean"]
            center_y=min(max_height/2,d["height"]*1.1)
            plt.plot(
                [center_x,d["ppm"]],
                [center_y,d["height"]/2],
                "k",alpha=0.3,linewidth=1)
            plt.text(center_x,center_y,d["species"],rotation="vertical",ha="center",alpha=0.6)

        plt.plot(df["ppm_median"], df["height"], "+", label="peaks median")
        plt.plot(df["ppm_max"], df["height"], "+", label="peaks maximum")
        plt.plot(df["ppm_mean"], df["height"], "+", label="peaks mean")

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

    try:
        df['startTime'] = dateutil.parser.parse(data_dict['acqu']['startTime'])
    except KeyError:
        df['startTime'] = dateutil.parser.parse("01.01.1990")



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


def main(folder):
    print(folder)
    results_df = pd.DataFrame()
    nindx = ["path", 'Sample', 'startTime',"delta_time", "peak"]
    snindx = ["path", 'Sample', 'startTime', "peak"]
    if RESULT_TABLE:
        res_file = os.path.join(folder, "results.xlsx")
        res_file_flat = os.path.join(folder, "results_flat.xlsx")
        try:
            results_df = pd.read_excel(res_file, index_col=list(range(len(nindx))))
        except FileNotFoundError:
            pass
    change = False

    sresults_df = results_df.copy()
    results_df = results_df.reset_index()
    def _sp(path):
        iname=os.path.basename(path)
        #while iname.startswith("0") and len(iname)>1:
        #    iname=iname[1:]
        #print(iname)
        #if iname.startswith("0") and int(iname)<100:
        #    return True
        return "path" in sresults_df.index.names and path in sresults_df.index.get_level_values(
            sresults_df.index.names.index('path')) and not RECREATE




    for data, data_dict in find_nmrs(folder, skip_path=_sp):
        path = data_dict["path"]
        try:
            sdf = work_spec(data, data_dict, path)
        except (PeakNotFoundError,IndexError) as e:
            continue

        sdf["path"] = path
        sdf["peak"] = sdf.index.values + 1


        for r,d in sdf.iterrows():
            sd=results_df
            for c in snindx:
                if c not in sd.columns:
                    continue
                sd=sd[sd[c]==d[c]]
            if len(sd)>0:
                results_df.loc[sd.index[0]]=d
            else:
                results_df = results_df.append(d)
        #sdf.set_index([nx for nx in nindx if nx in sdf.columns], inplace=True)

        #results_df = pd.concat([results_df, sdf[~sdf.index.isin(results_df.index)]])
        #results_df.update(sdf)
        change = True

    if change and RESULT_TABLE:
        results_df.sort_values(["startTime","ppm"],inplace=True)

        results_df["delta_time"]=(results_df["startTime"]-results_df.iloc[0]["startTime"])
        results_df["delta_time"] = results_df["delta_time"].apply(lambda d: d.total_seconds())

        try:
            results_df.to_excel(res_file_flat, merge_cells=True)
        except PermissionError:
            print(f"cannot write to file {res_file_flat}, maybe it is opened in another program?")

        try:
            results_df.set_index(nindx).to_excel(res_file, merge_cells=True)
        except PermissionError:
            print(f"cannot write to file {res_file}, maybe it is opened in another program?")
    else:
        print("no changes detected")

    def moving_average(a, n=3) :
        n=min(n,a.shape[0])
        a=a.copy()
        if len(a.shape)==1:
            a=a.reshape(-1,1)


        ret = np.cumsum(np.nan_to_num(a), dtype=float,axis=0)
        starts = (ret!=0).argmax(0)

        starts[ret[starts,np.arange(starts.shape[0])]==0]=a.shape[0]-1
        for i,s in enumerate(starts):
            a[:s,i]=a[s,i]

        ret = np.cumsum(np.nan_to_num(a), dtype=float,axis=0)

        ret[n:] = ret[n:] - ret[:-n]
        ret[n - 1:]/=n
        if n>1:
            ret[:n - 1]=(ret[:n - 1].T/np.arange(1,n)).T
        for i,s in enumerate(starts):
            ret[s:s+n,i]=[np.nanmean(a[s:s+_n,i]) for _n in range(1,n+1)]
            ret[:s,i]=np.nan
        return ret


    delta_times=results_df["delta_time"].unique()
    species_peaks=results_df["species_peak"].unique()
    species=results_df["species"].unique().astype(str)
    species_peaks=species_peaks[~np.isnan(species_peaks)]
    species=species[(~(species=="nan"))&(~(species==""))]

    mol_wts=np.array([SPECIES_MW[SPECIES_PEAKS_NAMES.index(n)] for n in species])

    plt.figure()
    for p in species_peaks:
        spd = results_df[results_df["species_peak"]==p]
        time_dict={d:np.nan for d in delta_times}
        for r,k in spd.iterrows():
            time_dict[k["delta_time"]]=k["area"]
        (keys,values) = zip(*time_dict.items())
        keys,values = np.array(keys),np.array(values)
        values=values[np.argsort(keys)]

        plt.plot(spd["delta_time"], spd["area"], ".", label=f"{p} ppm")
        t=40
        #mvgav1 = moving_average(values,t1)
        #mvgav1 = moving_average(mvgav1,t2)
        mvgav = moving_average(values,t)
        #mvgav[:-int(t/2)]=mvgav[int(t/2):]
        #plt.plot(delta_times, mvgav1, "-", label=f"avg. {p} ppm")
        plt.plot(delta_times, mvgav, "-", label=f"avg. {p} ppm")
        #plt.plot(delta_times, a, "-", label=f"a. {p} ppm")

    plt.legend()
    plt.title("NMR areas area over time")
    plt.xlabel("t [s]")
    plt.ylabel("rel. area")
    plt.savefig(os.path.join(folder, "nmr_area_over_time.png"),dpi=300)
    plt.close()

    plt.figure()
    for p in species:
        spd = results_df[results_df["species"]==p]
        time_dict={d:(np.nan,0) for d in delta_times}
        for r,k in spd.iterrows():
            time_dict[k["delta_time"]]=(k["estimated_rel_conc"],1) if time_dict[k["delta_time"]][1]==0 else (time_dict[k["delta_time"]][0]+k["estimated_rel_conc"],time_dict[k["delta_time"]][1]+1)

        for k in time_dict.keys():
            time_dict[k]=time_dict[k][0]/time_dict[k][1] if time_dict[k][1]>0 else time_dict[k][0]

        (keys,values) = zip(*time_dict.items())
        keys,values = np.array(keys),np.array(values)
        values=values[np.argsort(keys)]

        plt.plot(spd["delta_time"], spd["estimated_rel_conc"], ".", label=f"{p} ppm")
        t=40
        #mvgav1 = moving_average(values,t1)
        #mvgav1 = moving_average(mvgav1,t2)
        mvgav = moving_average(values,t)
        mvgav[:-int(t/2)]=mvgav[int(t/2):]
        #plt.plot(delta_times, mvgav1, "-", label=f"avg. {p} ppm")
        plt.plot(delta_times, mvgav, "-", label=f"avg. {p} ppm")
        #plt.plot(delta_times, a, "-", label=f"a. {p} ppm")

    plt.legend()
    plt.title("Species conc over time")
    plt.xlabel("t [s]")
    plt.ylabel("rel. conc")
    plt.savefig(os.path.join(folder, "species_over_time.png"),dpi=300)
    plt.close()



    species_weight=np.zeros((len(delta_times),len(mol_wts)))
    for i,p in enumerate(species):
        spd = results_df[results_df["species"]==p]
        time_dict={d:(np.nan,0) for d in delta_times}
        for r,k in spd.iterrows():
            time_dict[k["delta_time"]]=(k["estimated_rel_conc"],1) if time_dict[k["delta_time"]][1]==0 else (time_dict[k["delta_time"]][0]+k["estimated_rel_conc"],time_dict[k["delta_time"]][1]+1)

        for k in time_dict.keys():
            time_dict[k]=time_dict[k][0]/time_dict[k][1] if time_dict[k][1]>0 else time_dict[k][0]

        (keys,values) = zip(*time_dict.items())
        keys,values = np.array(keys),np.array(values)
        sorter=np.argsort(keys)
        values=values[sorter]
        keys=keys[sorter]
        mol_wt=SPECIES_MW[SPECIES_PEAKS_NAMES.index(p)]
        species_weight[np.argsort(keys),i]=values*mol_wt
    species_weight=100*(species_weight.T/np.nan_to_num(species_weight).sum(1)).T
    plt.figure()
    for i,p in enumerate(species):
        plt.plot(delta_times, species_weight[:,i], ".", label=f"{p} ppm")
    plt.legend()
    plt.title("w% over time")
    plt.xlabel("t [s]")
    plt.ylabel("w%")
    plt.savefig(os.path.join(folder, "wp_over_time.png"),dpi=300)
    plt.close()



    return
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
    #plt.savefig(os.path.join(folder, "nmr_area_over_time.png"),dpi=300)
    #plt.show()
    #plt.close()

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
    #plt.savefig(os.path.join(folder, "species_conc.png"),dpi=300)
    #plt.show()
    #plt.close()

    def ffill(arr):
        arr=arr.copy()
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        arr = arr[idx]
        return arr



    os.makedirs(os.path.join(folder,"timelaps",),exist_ok=True)


    for ti in range(2,csa.shape[0],1):
        cs=csa[:ti]
        tdiff=tdiffa[:ti]
        impath=os.path.join(folder,"timelaps", f"{ti:06d}.png")
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

        m=np.nanmax(entry_point)
        n=np.nanmin(entry_point)
        if ep_idx>0:
            ax2.vlines(tdiff[ep_idx],n,m,color="green",label=f"rxn start ({tdiff[ep_idx]})")


            if lp_idx>ep_idx:
                ax2.vlines(tdiff[lp_idx],n,m,color="red",label=f"rxn end ({tdiff[lp_idx]})")

        ax1.legend(loc='upper right')
        ax2.legend()
        ax2.set_xlim(*ax1.get_xlim())
        plt.savefig(impath,dpi=200)
        #plt.show()
        plt.close()

    return
    images=[]
    import imageio
    for f in sorted(os.listdir(os.path.join(folder,"timelaps"))):
        images.append(imageio.imread(os.path.join(folder,"timelaps",f)))
    imageio.mimsave(os.path.join(folder,"timelaps.gif"), images)

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
    for f in FOLDER:
        main(f)
