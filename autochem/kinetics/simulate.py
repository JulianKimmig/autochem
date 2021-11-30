import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def simulate_reaction_set(reaction_matrix,reaction_rates,initial_concentrations,times,constants=None):
    if constants is None:
        constants=np.zeros_like(initial_concentrations).astype(bool)

    assert reaction_matrix.ndim == 3
    assert reaction_rates.ndim == 2
    assert initial_concentrations.ndim == 1

    assert reaction_matrix.shape[1]==2
    assert reaction_rates.shape[1]==2

    assert reaction_rates.shape[0] == reaction_matrix.shape[0]
    assert initial_concentrations.shape[0] == reaction_matrix.shape[2]

    assert times.ndim == 1
    assert times.shape[0]>=2

    assert constants.ndim==1
    assert constants.shape[0]==initial_concentrations.shape[0]

    reaction_indices=np.arange(reaction_matrix.shape[0])
    reactand_indices=np.arange(reaction_matrix.shape[2])
    ln_reaction_rates=np.log(reaction_rates)

    constant_fac=(~constants).astype(int)

    def fun(t,y):
        dc=np.zeros_like(y)
        log_y=np.log(y)
        #print("log_y",log_y)

        for reac_idx in reaction_indices:
            reaction=reaction_matrix[reac_idx]
            ln_rate=ln_reaction_rates[reac_idx]

            #forward
            v_fwd=ln_rate[0]+np.nan_to_num(reaction[0]*log_y).sum()
            v_fwd=np.exp(v_fwd)
            dc-=v_fwd*reaction[0]
            dc+=v_fwd*reaction[1]
            #for r_idx in reactand_indices:
            #    c[i]+=

            #backwards
            v_bwd=ln_rate[1]+np.nan_to_num(reaction[1]*log_y).sum()
            v_bwd=np.exp(v_bwd)
            dc-=v_bwd*reaction[1]
            dc+=v_bwd*reaction[0]
            #print(v_fwd,v_bwd)
            #print(reaction[0]*log_y,reaction[1]*log_y)
            #print(np.nan_to_num(reaction[0]*log_y),np.nan_to_num(reaction[1]*log_y))
            #break

        dc*=constant_fac




        return dc

    if times.shape[0]>2:
        t_eval=times
    else:
        t_eval=None
    return solve_ivp(fun, (times.min(),times.max()), initial_concentrations,t_eval=t_eval)

if __name__ == '__main__':
    rm=[
        [[1,1,0,0,0,0,0],[0,0,1,0,0,0,0]],
        [[1,0,1,0,0,0,0],[0,0,0,1,0,0,0]],
        [[1,0,0,1,0,0,0],[0,0,0,0,1,0,0]],
    ]
    rc=[
        [0.5,0],
        [0.5,0],
        [0.5,0],
    ]
    ic=[3,7,0,0,0,0,0]
    rm=np.array(rm)
    rc=np.array(rc)
    ic=np.array(ic)
    times = np.linspace(0,5,20)
    r=simulate_reaction_set(rm, rc, ic, times)
    for c in  r["y"]:
        plt.plot(r["t"],c)
    plt.show()
    plt.close()


    rm=[
        [[1,1,0,0],[0,0,1,0]],
        [[0,0,1,0],[0,0,0,1]],
    ]
    rc=[
        [4,0.04],
        [0.05,0],
    ]
    ic=[0.01,0.01,0,0]
    rm=np.array(rm)
    rc=np.array(rc)
    ic=np.array(ic)
    times = np.linspace(0,100,100)
    r=simulate_reaction_set(rm, rc, ic, times,constants=np.array([False,False,False,False]))
    for c in  r["y"]:
        plt.plot(r["t"],c)
    plt.show()
    plt.close()