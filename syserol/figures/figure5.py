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
    ax[0].tick_params(axis='both', which='major', labelsize=10)
    ax[0].legend(fontsize='medium')
    ax[0].set_ylabel("Component Weight", fontsize=11.5)
    ax[0].set_xlabel("Time (days)", fontsize=11.5)

    # add some missingness
    imputeSim(sim_tensor, 0.5)
    # run factorization
    tFac = perform_contTF(sim_tensor, r=4)
    # plot factorization curves 
    days = dayLabels()
    lineplot(tFac, days.astype(int), "Time (days)", ax[1])
    ax[1].text(27, .4, f"Corrindex: {round(correlation_index(tFac.factors, sim_factors.factors), 3)}", 
        bbox=dict(boxstyle='square', fc="w", ec="k"))
    ax[1].set_title(rf"Factorization with 50% of Values Removed", fontsize='x-large')
    ax[1].tick_params(axis='both', which='major', labelsize=10)


    np.random.seed(1234)
    noise = np.random.normal(size=sim_tensor.shape)
    # scale the noise to have the same std of original tensor, so that when we add 1, 10, ..., 10,000 * noise it makes sense
    noise *= np.std(copy) 
    # have to use copy because impute changes in place
    noisyTensor = copy + noise
    tFac_noisy = perform_contTF(noisyTensor, r=4)
    lineplot(tFac_noisy, days.astype(int), "Time (days)", ax[2])
    ax[2].text(27, .4, f"Corrindex: {round(correlation_index(tFac_noisy.factors, sim_factors.factors), 3)}", 
        bbox=dict(boxstyle='square', fc="w", ec="k"))
    ax[2].set_title("Factorization with Noise Added", fontsize='x-large')
    ax[2].tick_params(axis='both', which='major', labelsize=10)
    
    

    iters = 5
    # Increase missingness and analyze the correlation index
    missInterval = [.5, .55, .6, .65, .7, .75, .8, .83, .87, .9, .93, .96, .99]
    corrindex_miss = np.zeros((iters, len(missInterval)))
    for i in range(iters):
        for j, miss in enumerate(missInterval):
            sim_tensor, _, sim_factors = generate_simulated()
            imputeSim(sim_tensor, miss)
            tFac = perform_contTF(sim_tensor, r=4)
            corrindex_miss[i, j] = correlation_index(sim_factors.factors, tFac.factors)

    print("Missing iters: ", corrindex_miss)

    # Vary noise scale and check correlation index
    scale = np.array([0.1, 1, 10, 100, 1000, 10000])
    corrindex_noise = np.zeros((iters, len(scale)))
    np.random.seed()
    for iter in range(iters):
        sim_tensor, _, sim_factors = generate_simulated()
        noise = np.random.normal(size=sim_tensor.shape)
        noise *= np.std(sim_tensor)
        for idx, size in enumerate(scale):
            noisyTensor = sim_tensor + noise*size
            tFac_noisy = perform_contTF(noisyTensor, r=4)
            corrindex_noise[iter, idx] = correlation_index(sim_factors.factors, tFac_noisy.factors)

    print("Noise iters: ", corrindex_noise)

    ax[3].errorbar(missInterval, corrindex_miss.mean(axis=0), corrindex_miss.std(axis=0), linestyle='None', marker='o', ms=4)
    ax[3].set_ylabel("Correlation Index", fontsize=11.5)
    ax[3].set_xlabel("Missingness Percentage", fontsize=11.5)
    ax[3].set_ylim(bottom=0.0)
    ax[3].tick_params(axis='both', which='major', labelsize=10)
    
    
    ax[4].errorbar(scale, corrindex_noise.mean(axis=0), corrindex_noise.std(axis=0), linestyle='None', marker='o', ms=4)
    ax[4].set_ylabel("Correlation Index", fontsize=11.5)
    ax[4].set_xlabel("Noise Scalar", fontsize=11.5)
    ax[4].set_xticks([x for x in scale])
    ax[4].set_xticklabels([x for x in scale])
    ax[4].set_xscale("log")
    ax[4].set_ylim(bottom=0.0)
    ax[4].tick_params(axis='both', which='major', labelsize=10)
    


    subplotLabel(ax)
    return f

