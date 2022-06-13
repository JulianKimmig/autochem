from typing import Dict,List
import logging
import os, sys
import numpy as np
if __name__ == '__main__':
    ddir = os.path.dirname(os.path.abspath(__file__))
    while "autochem" not in os.listdir(ddir):
        ddir = os.path.dirname(ddir)
    sys.path.insert(0, ddir)
    sys.path.append(ddir)

    logging.basicConfig()


logger = logging.getLogger("autochem")
logger.setLevel("DEBUG")
from collections import defaultdict
import matplotlib.pyplot as plt
from autochem.spectra.nmr.reader import read_nmr, NMRReadError
from autochem.utils.signals.peak_detection import find_peaks, cut_peaks_data, peak_integration, manual_peak_finder
from autochem.spectra.nmr.utils import zoom
import pandas as pd
import dateutil

def solve_preprocess(preprocess):
    if isinstance(preprocess,str):
        if preprocess == "normalization":
            from autochem.utils.corrections import norm_data
            return lambda data: norm_data(data)
    if isinstance(preprocess,dict):
        assert len(preprocess) == 1, "only one preprocess action at a time"
        action=list(preprocess.keys())[0]
        data=list(preprocess.values())[0]

        if action == "baseline_correction":
            if data == "median":
                from autochem.utils.corrections.baseline import median_correction
                return lambda data: median_correction(data)
            else:
                raise NotImplementedError(f"baseline correction with {data} not implemented")
        else:
            raise NotImplementedError(f"{action} not implemented")
    else:
        raise ValueError(f"preprocess '{preprocess}' not valid")

def process_spectra(data:np.array,udic:Dict,data_processing:Dict,plotting:Dict,peak_picking:List,path:str,cutoff_ouside_data=True):
    _plot_idx=3
    _plot_number=10**(-_plot_idx)
    _plot_number=np.round(_plot_number,_plot_idx)
    _plt_fwd=_plot_number
    
    ppm_scale= udic['ppm_scale']

    if plotting['raw_data']:
        plt.plot(ppm_scale[::-1],data[::-1])
        plt.title("raw data")
        plt.savefig(os.path.join(path,f"{_plot_number}.png"))
        _plot_number=np.round(_plot_number+_plt_fwd,_plot_idx)
    preprocesses=[]
    for process in data_processing:
        action=solve_preprocess(process)
        data,_process=action(data)
        preprocesses.append(_process)

    if plotting['processed_data']:
        plt.plot(ppm_scale[::-1],data[::-1])
        plt.title("Processed data")
        plt.savefig(os.path.join(path,f"{_plot_number}.png"))
        _plot_number=np.round(_plot_number+_plt_fwd,_plot_idx)


    if "ranges" in peak_picking:
        peaks, peak_data = manual_peak_finder(y=data, x=ppm_scale,peak_ranges=peak_picking["ranges"], min_peak_height=peak_picking.get("min_peak_height",0.02),
                                  rel_height=peak_picking.get("rel_height",0.03),
                                  max_width=peak_picking.get("max_peak_width",np.inf),
                                  rel_prominence=peak_picking.get("rel_prominence",0.1),
                                  center="median",
                                  
                                  )
    else:
        peaks, peak_data = find_peaks(y=data, x=ppm_scale, min_peak_height=peak_picking.get("min_peak_height",0.02),
                                  rel_height=peak_picking.get("rel_height",0.03),
                                  max_width=peak_picking.get("max_peak_width",np.inf),
                                  min_distance=peak_picking.get("min_distance",0.01),
                                  rel_prominence=peak_picking.get("rel_prominence",0.1),
                                  center="median"
                                  )


    
    if cutoff_ouside_data:
        #zoom to relevant areas
        ppm_min = ppm_scale[peak_data["peak_left_border"].min()]
        ppm_max = ppm_scale[peak_data["peak_right_border"].max()]
        ppm_min,ppm_max=ppm_min-0.1*(ppm_max-ppm_min),ppm_max+0.1*(ppm_max-ppm_min)

        data, ppm_scale,zoom_indices = zoom(
        data,
        ppm_scale,
        ppm_min,
        ppm_max,
        )

        peaks, peak_data = cut_peaks_data(peaks, peak_data,*zoom_indices)

        if plotting['zoomed_data']:
            plt.plot(ppm_scale[::-1],data[::-1])
            plt.title("Zoomed data")
            plt.savefig(os.path.join(path,f"{_plot_number}.png"))
            _plot_number=np.round(_plot_number+_plt_fwd,_plot_idx)


    #integrate_peaks
    peaks, peak_data = peak_integration(x=ppm_scale,
                                        y=data,
                                        peaks=peaks,
                                        peak_data=peak_data,
                                        )

    if plotting['result']:
        plt.plot(ppm_scale[::-1],data[::-1])
        plt.scatter(ppm_scale[peaks],data[peaks],c='r',marker='+')

        print(peaks)
        for i in range(peaks.shape[0]):
            lb = peak_data['peak_left_border'][i]
            rb = peak_data['peak_right_border'][i]
            print(ppm_scale[lb],ppm_scale[rb])
            plt.fill_between(x=ppm_scale[lb:rb], y1=data[lb:rb], alpha=0.5,label=f"peak {i+1}")


        plt.title("Complete data")
        plt.savefig(os.path.join(path,f"{_plot_number}.png"))
        _plot_number=np.round(_plot_number+_plt_fwd,_plot_idx)

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
        df['startTime'] = dateutil.parser.parse(udic['acqu']['startTime'])
    except KeyError:
        df['startTime'] = dateutil.parser.parse("01.01.1990")

    df["Sample"] = udic['acqu'].get('Sample',"sample_name")

    df.to_excel(os.path.join(path,"signals_ac.xlsx"),merge_cells=True)
    return df


def main(path,data_processing:List=None,parse_subfolders=True,plotting:Dict[str,bool]=None,peak_picking:Dict=None,cutoff_ouside_data=True):
    if data_processing is None:
        data_processing = []
    if plotting is None:
        plotting = {}
    if peak_picking is None:
        peak_picking = {}

    plotting = defaultdict(lambda:False,plotting)

    try:
        data, udict = read_nmr(path, preprocess=True)
        logger.info("read '{}' as '{}'".format(path, udict["datatype"]))
        process_spectra(data,udict,data_processing,plotting,peak_picking,path,cutoff_ouside_data=cutoff_ouside_data)
    except NMRReadError as e:
        pass

    if parse_subfolders:
        for subpath, folder, files in os.walk(path):
            if subpath == path:
                continue
            try:
                data, udict = read_nmr(subpath, preprocess=True)
                logger.info("read '{}' as '{}'".format(subpath, udict["datatype"]))
                process_spectra(data,udict,data_processing,plotting,peak_picking,subpath,cutoff_ouside_data=cutoff_ouside_data)
            except NMRReadError as e:
                pass


    pass

if __name__ == '__main__':
    # config and path from argument
    import argparse
    parser = argparse.ArgumentParser(description='NMR workflow')
    parser.add_argument('-c', '--config', help='config file', required=True)
    parser.add_argument('-p', '--path', help='path to data', required=True)
    args = parser.parse_args()
    config_file = args.config
    path = args.path
    #laod config from yaml
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    

    main(path,**config)