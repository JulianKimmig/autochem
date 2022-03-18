from optuna.distributions import UniformDistribution, CategoricalDistribution, IntUniformDistribution, \
    DiscreteUniformDistribution
from optuna.samplers import RandomSampler

x_columns=["DP_CEAm","DP_GPAm","ratio1"]
y_column="EFGF"

x_distributions={
    "DP_CEAm":CategoricalDistribution([59,87,0]),#UniformDistribution(0,120),#
    "DP_GPAm":CategoricalDistribution([60,92,0]),#UniformDistribution(0,120),#
    "ratio1":DiscreteUniformDistribution(0,1,0.01),
}

def df_sample():
    d = []
    sampler = RandomSampler()
    for x in x_columns:
        d.append(sampler.sample_independent(None,None,x,x_distributions[x]))
    d.append(0)
    if d[x_columns.index("DP_GPAm")]== 0 or d[x_columns.index("ratio1")]== 0:
        return tuple(d)
    return None

maximum_modification_samples=100*3 + 1*3 # (possible ratios)*(possible DP_CEAm) + (ratio==0)

data_modification= df_sample

name="dp_ratio_dep"

datafile="own_table2.xlsx"
sheetname="data"