import imp
from re import sub
import seaborn as sns
from syserol.simulated import generate_simulated, imputeSim, viewP
from syserol.tensor import perform_CMTF
from syserol.COVID import dayLabels
from syserol.figures.common import getSetup, subplotLabel
from syserol.figures.figure2 import lineplot

def makeFigure():
    ax, f = getSetup((7,5), (1,2))
    # generate simulated data
    sim_tensor, P = generate_simulated()
    # view original P curves
    df = viewP(P)
    sns.lineplot(data=df, dashes=False, ax=ax[0])

    # add some missingness
    imputeSim(sim_tensor, 0.5)
    # run factorization
    tFac = perform_CMTF(sim_tensor)

    # plot factorization curves 
    days = dayLabels()
    lineplot(tFac, days.astype(int), "Time (days)", ax[1])

    subplotLabel(ax)
    return f

