import numpy as np
from matplotlib import pyplot as plt

from autochem.kinetics.simulate import simulate_reaction_set


def sim_ome():
    n_omes=10
    names=["TriOx","FA"]+[f"OME{n}" for n in range(1,n_omes+1)]



    rm=[
        [[1,0]+[0 for _ in  range(n_omes)],[0,3]+[0 for _ in  range(n_omes)]], #Triox->3 FA
    ]
    for j in range(n_omes-1): # -1 since last ome has no product
        rm.append(
            [[0,1]+[1 if i==j else 0 for i in  range(n_omes)],[0,0]+[1 if i==j+1 else 0 for i in  range(n_omes)]]
        )
    rc=[
        [0.01,0],
    ]+[[1,0.01] for  _ in  range(n_omes-1)]

    ic=[1,0,3]+[0 for _ in  range(n_omes-1)]
    rm=np.array(rm)
    rc=np.array(rc)
    ic=np.array(ic)
    print(rm.shape,rc.shape,ic.shape,)
    times = np.linspace(0,500,100)
    r=simulate_reaction_set(rm, rc, ic, times)
    for i,c in  enumerate(r["y"]):
        plt.plot(r["t"],c,label=names[i])
    plt.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    sim_ome()