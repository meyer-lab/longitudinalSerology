import imp
import numpy as np
from re import sub
import seaborn as sns
from syserol.simulated import generate_simulated, imputeSim, viewP
from syserol.tensor import perform_CMTF
from syserol.COVID import dayLabels
from syserol.figures.common import getSetup, subplotLabel
from syserol.figures.figure2 import lineplot

def makeFigure():
    ax, f = getSetup((8,5), (1,3))
    # generate simulated data
    sim_tensor, P = generate_simulated()
    copy = np.copy(sim_tensor)
    # view original P curves
    df = viewP(P)
    sns.lineplot(data=df, dashes=False, ax=ax[0])

    # add some missingness
    imputeSim(sim_tensor, 0.5)
    # run factorization
    tFac = perform_CMTF(sim_tensor, r=4)

    # plot factorization curves 
    days = dayLabels()
    lineplot(tFac, days.astype(int), "Time (days)", ax[1])

    noise = np.random.normal(size=sim_tensor.shape)
    # have to use copy because impute changes in place
    noisyTensor = copy + noise
    tFac_noisy = perform_CMTF(noisyTensor, r=4)
    lineplot(tFac_noisy, days.astype(int), "Time (days)", ax[2])

    subplotLabel(ax)
    return f

