import imp
import numpy as np
from re import sub
import seaborn as sns
from syserol.simulated import generate_simulated, imputeSim, viewP
from syserol.tensor import perform_contTF
from syserol.COVID import dayLabels
from syserol.figures.common import getSetup, subplotLabel
from syserol.figures.figure2 import lineplot
from tensorly.metrics import correlation_index

def makeFigure():
    ax, f = getSetup((10,7), (2,3))
    # generate simulated data
    sim_tensor, P, sim_factors = generate_simulated()
    copy = np.copy(sim_tensor)
    # view original P curves
    df = viewP(P)
    sns.lineplot(data=df, dashes=False, ax=ax[0])

    # add some missingness
    imputeSim(sim_tensor, 0.5)
    # run factorization
    tFac = perform_contTF(sim_tensor, r=4)

    # plot factorization curves 
    days = dayLabels()
    lineplot(tFac, days.astype(int), "Time (days)", ax[1])
    ax[1].text(27, .4, f"Corrindex: {round(correlation_index(tFac.factors, sim_factors.factors), 2)}", 
        bbox=dict(boxstyle='square', fc="w", ec="k"))


    np.random.seed(1234)
    noise = np.random.normal(size=sim_tensor.shape)
    # scale the noise to have the same std of original tensor, so that when we add 1, 10, ..., 10,000 * noise it makes sense
    noise *= np.std(copy) 
    # have to use copy because impute changes in place
    noisyTensor = copy + noise
    tFac_noisy = perform_contTF(noisyTensor, r=4)
    lineplot(tFac_noisy, days.astype(int), "Time (days)", ax[2])
    ax[2].text(27, .4, f"Corrindex: {round(correlation_index(tFac_noisy.factors, sim_factors.factors), 2)}", 
        bbox=dict(boxstyle='square', fc="w", ec="k"))


    # Increase missingness and analyze the correlation index
    missInterval = 15
    missingness = np.zeros(missInterval)
    corrindex_miss = np.zeros((5, missInterval))
    # start at baseline 50% missingness and go up from there. We modified to 50% missingness earlier.
    for i in range(missInterval):
        for iter in range(5):
            # run the factorization on this level of missingness 3 times, to get average for plot
            tFac = perform_contTF(sim_tensor, r=4)
            corrindex_miss[iter, i] = correlation_index(sim_factors.factors, tFac.factors)
        missingness[i] = (np.sum(np.isnan(sim_tensor)))/(np.size(sim_tensor))
        imputeSim(sim_tensor, 0.2) # add more missingness for next loop. The last time won't matter.

    # Vary noise scale and check correlation index
    scale = np.array([0.1, 1, 10, 100, 1000, 10000])

    corrindex_noise = np.zeros((3, len(scale)))
    for iter in range(3):
        np.random.seed()
        noise = np.random.normal(size=sim_tensor.shape)
        noise *= np.std(copy)
        for idx, size in enumerate(scale):
            noisyTensor = copy + noise*size
            tFac_noisy = perform_contTF(noisyTensor, r=4)
            corrindex_noise[iter, idx] = correlation_index(sim_factors.factors, tFac_noisy.factors)

    ax[3].errorbar(missingness, corrindex_miss.mean(axis=0), corrindex_miss.std(axis=0), linestyle='None', marker='o', ms=3)
    ax[3].set_ylabel("Correlation Index")
    ax[3].set_xlabel("Missingness Percentage")
    ax[3].set_ylim(bottom=0.0)
    
    ax[4].errorbar(scale, corrindex_noise.mean(axis=0), corrindex_noise.std(axis=0), linestyle='None', marker='o', ms=3)
    ax[4].set_ylabel("Correlation Index")
    ax[4].set_xlabel("Noise Scalar")
    ax[4].set_xscale("log")
    ax[4].set_ylim(bottom=0.0)


    subplotLabel(ax)
    return f

