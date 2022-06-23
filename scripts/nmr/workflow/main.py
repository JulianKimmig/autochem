from cProfile import label
from tkinter import N
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
from autochem.utils.signals.peak_detection import find_peaks, cut_peaks_data, peak_integration, manual_peak_finder,get_reference_peak,PeakNotFoundError,factorize_peak_data
from autochem.spectra.nmr.utils import zoom
import pandas as pd
import dateutil

def solve_preprocess(preprocess):
    if isinstance(preprocess,str):
        if preprocess == "normalization":
            from autochem.utils.corrections import norm_data
            return lambda x,y: norm_data(x,y)
    if isinstance(preprocess,dict):
        assert len(preprocess) == 1, "only one preprocess action at a time"
        action=list(preprocess.keys())[0]
        data=list(preprocess.values())[0]

        if action == "baseline_correction":
            if data == "median":
                from autochem.utils.corrections.baseline import median_correction
                return lambda x,y: median_correction(x,y)
            else:
                raise NotImplementedError(f"baseline correction with {data} not implemented")
        elif action=="auto_shift":
            from autochem.utils.corrections.shift import get_signal_shift
            targest=np.array(data["targets"])
            allow_stretch=data.get("allow_stretch",False)
            n=int(data.get("n",len(targest)*1.5))
            peak_find_kwargs=data.get("peak_find_kwargs",{})
            def _fp(x,y):
                peaks, peak_data = find_peaks(y=y, x=x,**peak_find_kwargs)
                peak_ppm=x[peaks]
                dist_matrix = np.abs( np.subtract.outer(peak_ppm,targest))
                peak_connection=np.zeros((len(targest),2))*np.nan
                for i in range(peak_connection.shape[0]):
                    if np.isnan(dist_matrix).all():
                        break

                    minidx = np.unravel_index(np.nanargmin(dist_matrix), dist_matrix.shape)
                    dist_matrix[minidx[0],:]=np.nan
                    dist_matrix[:,minidx[1]]=np.nan
                    peak_connection[i,:]=minidx
                peak_connection=peak_connection[~np.isnan(peak_connection).all(axis=1)].astype(int)

                #heights=y[peaks]
                #sorter = np.argsort(heights)[::-1]
                #print(x[peaks[sorter[:n]]])
                _m,_c = get_signal_shift(
                    #x[peaks[sorter[:n]]],
                    peak_ppm[peak_connection[:,0]],
                    targest,allow_stretch=allow_stretch)
                x=x*_m + _c
                return x,y,[_m,_c]
            return _fp
        elif action=="regulize":
            ppu = data.get('ppu',100)
            def _reg(x,y):
                xn=np.linspace(x.min(),x.max(),int(np.ceil((x.max()-x.min())*ppu)))
                yn=np.interp(xn,x,y)
                return xn,yn,None
            return _reg
        else:
            raise NotImplementedError(f"{action} not implemented")
    else:
        raise ValueError(f"preprocess '{preprocess}' not valid")

def udict_time(udict):
    try:
        return dateutil.parser.parse(udict['acqu']['startTime'])
    except KeyError:
        return dateutil.parser.parse("01.01.1990")

def process_spectra(data:np.array,udic:Dict,data_processing:Dict,plotting:Dict,peak_picking:List,path:str,cutoff_ouside_data=True,reference=None,**kwargs):
    _plot_idx=3
    _plot_number=10**(-_plot_idx)
    _plot_number=np.round(_plot_number,_plot_idx)
    _plt_fwd=_plot_number
    
    ppm_scale= udic['ppm_scale']

    if plotting['raw_data']:
        plt.plot(ppm_scale,data)
        plt.gca().invert_xaxis()
        plt.title("raw data")
        plt.savefig(os.path.join(path,f"{_plot_number}.png"))
        plt.close()
        _plot_number=np.round(_plot_number+_plt_fwd,_plot_idx)
    preprocesses=[]
    for process in data_processing:
        action=solve_preprocess(process)
        ppm_scale,data,_process=action(ppm_scale,data)
        preprocesses.append(_process)

    if plotting['processed_data']:
        plt.plot(ppm_scale,data)
        plt.gca().invert_xaxis()
        plt.title("Processed data")
        plt.savefig(os.path.join(path,f"{_plot_number}.png"))
        plt.close()
        _plot_number=np.round(_plot_number+_plt_fwd,_plot_idx)

    peak_ids=None
    if "ranges" in peak_picking:
        peaks, peak_data = manual_peak_finder(y=data, x=ppm_scale,peak_ranges=peak_picking["ranges"],
                                  #min_peak_height=peak_picking.get("min_peak_height",0.02),
                                  #rel_height=peak_picking.get("rel_height",0.03),
                                  #max_width=peak_picking.get("max_peak_width",np.inf),
                                  #rel_prominence=peak_picking.get("rel_prominence",0.1),
                                  center="max",
                                  
                                  )          
        assert len(peaks) == len(peak_picking["ranges"])

    else:

        peaks, peak_data = find_peaks(y=data, x=ppm_scale, min_peak_height=peak_picking.get("min_peak_height",peak_picking.get("min_height",0.02)),
                                  rel_height=peak_picking.get("rel_height",0.03),
                                  max_width=peak_picking.get("max_peak_width",np.inf),
                                  min_distance=peak_picking.get("min_distance",0.01),
                                  rel_prominence=peak_picking.get("rel_prominence",0.1),
                                  center="max"
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
            plt.plot(ppm_scale,data)
            plt.gca().invert_xaxis()
            plt.title("Zoomed data")
            plt.savefig(os.path.join(path,f"{_plot_number}.png"))
            plt.close()
            _plot_number=np.round(_plot_number+_plt_fwd,_plot_idx)



    #integrate_peaks
    peaks, peak_data = peak_integration(x=ppm_scale,
                                        y=data,
                                        peaks=peaks,
                                        peak_data=peak_data,
                                        )

    if reference is not None:
        pidx=None
        
        pidx, peak = get_reference_peak(
        ppm_scale[peaks],
        reference["ppm"],
        max_diff=reference.get("window",np.inf))
       

        if pidx is not None:
            scaler=reference.get("area",1) / peak_data["integrals"][pidx]
            peak_data = factorize_peak_data(peak_data,scale_factor = scaler)

    if plotting['result']:
        plt.plot(ppm_scale,data)
        plt.scatter(ppm_scale[peaks],data[peaks],c='r',marker='+')

        print(peaks)
        for i in range(peaks.shape[0]):
            lb = peak_data['peak_left_border'][i]
            rb = peak_data['peak_right_border'][i]
            print(ppm_scale[lb],ppm_scale[rb])
            plt.fill_between(x=ppm_scale[lb:rb], y1=data[lb:rb], alpha=0.5,label=f"peak {i+1}")


        plt.title("Complete data")
        plt.gca().invert_xaxis()
        plt.savefig(os.path.join(path,f"{_plot_number}.png"))
        plt.close()
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

    if "ranges" in peak_picking:
        ranges=np.array(peak_picking["ranges"])
        means=ranges.mean(1)
        mean_sort=np.argsort(means)
        ppm_sorm=np.argsort(ppm_scale[peaks])
        print(mean_sort,ppm_sorm)
        ranges=ranges[mean_sort][ppm_sorm]
        df["range"]=[f"{s}-{e}" for s,e in ranges]


    df['startTime'] = udict_time(udic)

    df["Sample"] = udic['acqu'].get('Sample',"sample_name")
    df["#"]=np.arange(len(df))+1
    df.to_excel(os.path.join(path,"signals_ac.xlsx"),merge_cells=True)
    return df,ppm_scale,data


def main(path,data_processing:List=None,parse_subfolders=True,plotting:Dict[str,bool]=None,peak_picking:Dict=None,cutoff_ouside_data=True,**kwargs):
    if data_processing is None:
        data_processing = []
    if plotting is None:
        plotting = {}
    if peak_picking is None:
        peak_picking = {}

    plotting = defaultdict(lambda:False,plotting)

    df = None
    if "time_series" in kwargs:
        ts_data={}
        ts_specs={}

    try:
        data, udict = read_nmr(path, preprocess=True)
        logger.info("read '{}' as '{}'".format(path, udict["datatype"]))
        df,ppm,data = process_spectra(data,udict,data_processing,plotting,peak_picking,path,cutoff_ouside_data=cutoff_ouside_data,**kwargs)
        df["path"]=os.path.relpath(path, path)
        if "time_series" in kwargs:
            ts_specs[udict_time(udict)]=(ppm,data)
    except NMRReadError as e:
        pass
    
    
  
    
    if parse_subfolders:
        for subpath, folder, files in os.walk(path):
            if subpath == path:
                continue
            #if os.path.basename(subpath).startswith("000") and  int(os.path.basename(subpath))<90:
            #    continue
            try:
                data, udict = read_nmr(subpath, preprocess=True)
                logger.info("read '{}' as '{}'".format(subpath, udict["datatype"]))
                _df,ppm,data = process_spectra(data,udict,data_processing,plotting,peak_picking,subpath,cutoff_ouside_data=cutoff_ouside_data,**kwargs)
                _df["path"]=os.path.relpath(subpath, path)
                if "time_series" in kwargs:
                    ts_specs[udict_time(udict)]=(ppm,data)

                if df is None:
                    df = _df
                else:
                     df = pd.concat([df,_df])
                #if len(df)>100:
                #    break
            except NMRReadError as e:
                pass
            except Exception as e:
                logger.exception(e)

    grouper=["path"]
    if "ranges" in peak_picking: 
        grouper.append("range")
    grouper.append("#")
    df.set_index(grouper, inplace=True)


    df.to_excel(os.path.join(path,"all_spectra.xlsx"))

    
    if "time_series" in kwargs:
        parameter=kwargs["time_series"]["parameter"]
        smooth_weight=kwargs["time_series"].get("smooth",0)
        excludes=kwargs["time_series"].get("excludes",{})
        
        excluded_times=set()

        from autochem.utils.corrections.smooth import smooth
        for r,d in df.iterrows():
            key=r[1]
            if key not in ts_data:
                ts_data[key]={}
            ts_data[key][d["startTime"]]=d[parameter]
            

        ts_df=pd.DataFrame(ts_data)
        SPECMAP_RES=1000
        ppm_min=np.inf
        ppm_max=-np.inf
        times=sorted(list(ts_specs.keys()))
        time_min=min(times)
        time_max=max(times)
        for time,(ppm,data) in ts_specs.items():
            ppm_min=min(ppm_min,ppm.min())
            ppm_max=max(ppm_max,ppm.max())

        spec_map=np.zeros((len(ts_specs),SPECMAP_RES))*np.nan
        global_ppm=np.linspace(ppm_min,ppm_max,SPECMAP_RES)
        ppppm=SPECMAP_RES/(ppm_max-ppm_min)

        for i,time in enumerate(times):
            if time not in ts_specs:
                continue
            ppm,data=ts_specs[time]
            spec_map[i,:]=np.interp(global_ppm,ppm,data)
        
        spec_map_change=np.zeros_like(spec_map)
        spec_map_change[1:,:]=spec_map[1:,:]-spec_map[:-1,:]


        plt.figure()
        plt.imshow(spec_map_change[::-1],aspect="auto",
        extent=[ppm_min,ppm_max,0,(time_max-time_min).total_seconds()/60,]
        )
        plt.colorbar()
        plt.ylabel("time")
        plt.xlabel("ppm")

        plt.savefig(os.path.join(path,"time_series_map_difference.png"),dpi=300)
        

        spect_change_sum=spec_map_change.sum(1)/ppppm
        plt.figure()
        plt.plot(times,spect_change_sum)
        plt.xlabel("time")
        plt.ylabel("change per ppm")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(path,"time_series_map_change_sum.png"),dpi=300)

        
            


        sum_max=excludes.get("sum_max",np.inf)
        sum_min=excludes.get("sum_min",-np.inf)

        sum_max_exclude=list(ts_df[ts_df.sum(1)>sum_max].index)
        excluded_times.update(sum_max_exclude)
        ts_df.drop(sum_max_exclude, inplace=True)
        
        sum_min_exclude=list(ts_df[ts_df.sum(1)<sum_min].index)
        excluded_times.update(sum_min_exclude)
        ts_df.drop(sum_min_exclude, inplace=True)

        rel_change_max=excludes.get("rel_change_max",np.inf)
        for i,c in enumerate(spect_change_sum):
            if c>rel_change_max:
                excluded_times.add(times[i])
                ts_df.drop(times[i], inplace=True)
                spec_map_change
                


        ts_df.sort_index(inplace=True)
        ts_df.to_excel(os.path.join(path,"time_series.xlsx"))
        plt.figure()
        alpha=0.3 if smooth_weight>0 else 1
        for c in ts_df.columns:
            line, = plt.plot(ts_df.index,ts_df[c],label=str(c),alpha=alpha)
            if smooth_weight>0:
                plt.plot(ts_df.index,smooth(ts_df[c],smooth_weight),label=f"{c} smooth",linestyle="-", color=line.get_color())
        
        plt.xlabel("time")
        plt.xticks(rotation=90)
        plt.ylabel("intensity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path,"time_series.png"),dpi=300)
        plt.close()

        tsd=os.path.join(path,"time_series_detailed")
        os.makedirs(tsd,exist_ok=True)
        for c1 in ts_df.columns:
            c1d=ts_df[c1].values
            for c2 in ts_df.columns:
                if c1==c2:
                    continue
                c2d=c1d/ts_df[c2].values

                plt.figure()
                line, = plt.plot(ts_df.index,c2d,label=str(c1),alpha=alpha)
                if smooth_weight>0:
                    plt.plot(ts_df.index,smooth(c2d,smooth_weight),label=f"{c1} smooth",linestyle="-", color=line.get_color())
                plt.title(f"{c1} with const. {c2}")
                plt.xlabel("time")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.ylabel("intensity")
                plt.legend()
                plt.savefig(os.path.join(tsd,f"{c1}_const_{c2}.png"),dpi=300)
                plt.close()

        for t in excluded_times:
            if t in ts_specs:
                del ts_specs[t]

        ppm_min=np.inf
        ppm_max=-np.inf
        time_min=min(times)
        time_max=max(times)

        single_spec_dir=os.path.join(path,"single_spectra")
        os.makedirs(single_spec_dir,exist_ok=True)

        for time,(ppm,data) in ts_specs.items():
            ppm_min=min(ppm_min,ppm.min())
            ppm_max=max(ppm_max,ppm.max())
        
        
        
        for i,time in enumerate(times):
            if time not in ts_specs:
                continue
            ppm,data=ts_specs[time]
            spec_map[-i-1,:]=np.interp(global_ppm,ppm,data)
            plt.figure()
            plt.plot(ppm,data)
            plt.savefig(os.path.join(single_spec_dir,f"{time.strftime('%y_%m_%d-%H_%M_%S')}.png"),dpi=300)
            plt.close()

        plt.figure()
        plt.imshow(spec_map,aspect="auto",
        extent=[ppm_min,ppm_max,0,(time_max-time_min).total_seconds()/60,]
        )
        plt.colorbar()
        plt.ylabel("time")
        plt.xlabel("ppm")

        plt.savefig(os.path.join(path,"time_series_map.png"),dpi=300)

        spec_map=np.log(spec_map)
        plt.imshow(spec_map,aspect="auto",
        extent=[ppm_min,ppm_max,0,(time_max-time_min).total_seconds()/60,]
        )
        plt.colorbar()
        plt.ylabel("time")
        plt.xlabel("ppm")

        plt.savefig(os.path.join(path,"time_series_map_log.png"),dpi=300)



    pass

if __name__ == '__main__':
    # config and path from argument
    import argparse
    parser = argparse.ArgumentParser(description='NMR workflow')
    parser.add_argument('-c', '--config', help='config file',default=None, required=False)
    parser.add_argument('-p', '--path', help='path to data', required=True)
    import shutil
    args = parser.parse_args()
    config_file = args.config
    path = args.path
    if config_file is None:
        config_file = os.path.join(path,"config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"config file '{config_file}' not found")
    
    #laod config from yaml
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if os.path.abspath(config_file)!=os.path.abspath( os.path.join(path,"config.yaml")):
        shutil.copy(config_file, os.path.join(path,"config.yaml"))

    main(path,**config)