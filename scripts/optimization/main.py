import importlib
import os

import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt
from optuna.distributions import UniformDistribution
from optuna.samplers import TPESampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator


def main(config_path):
    spec=importlib.util.spec_from_file_location("optimconfig",config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    os.chdir(os.path.dirname(config_path))

    datafile= config.datafile
    data=pd.read_excel(datafile,sheet_name=getattr(config,"sheetname",None))[config.x_columns+[config.y_column]]
    data.dropna(inplace=True)

    study = optuna.create_study(study_name=config.name,
                                direction="maximize",
                                sampler=TPESampler(n_ei_candidates=100)
                                )
    mod_d=set()
    while len(mod_d)<config.maximum_modification_samples:
        print(len(mod_d),end="\r")
        r=config.data_modification()
        if r is not None:
            mod_d.add(r)
    mod_d=sorted(list(mod_d),key=lambda x:x[2])

    sdf=pd.DataFrame(mod_d,columns=data.columns)
    data=data.append(sdf)
    for r,d in data.iterrows():
        try:
            trial = optuna.trial.create_trial(
                params={x:d[x] for x in config.x_columns},
                distributions={x:config.x_distributions[x] for x in config.x_columns},
                value=d[config.y_column],
            )

            study.add_trial(trial)
        except Exception as e:
            print(e)




    trial = study.ask(fixed_distributions={x:config.x_distributions[x] for x in config.x_columns})
    print(trial.params)

    trials=[]
    k=1000
    for i in range(k):
        trial = study.ask(fixed_distributions={x:config.x_distributions[x] for x in config.x_columns})
        trials.append(trial.params)
        #df.loc[len(df)]=trial.params
    df = pd.DataFrame(trials,columns=config.x_columns)

    for c in df.columns:
        plt.hist(df[c],bins=min(100,int(k/2)))
        plt.show()
        plt.close()


    fig = optuna.visualization.plot_contour(study)
    #fig = optuna.visualization.plot_contour(study, params=config.x_columns[1:3])
    #fig = optuna.visualization.plot_param_importances(study)
    fig.show()

   # print(data)



if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description="Optimization script")
    argparser.add_argument("--config", type=str,required=True, help="Path to the config file")

    args = argparser.parse_args()

    main(args.config)