import numpy as np
import seaborn as sns
from .common import getSetup, subplotLabel
from syserol.COVID import Tensor4D, dayLabels, dimensionLabel3D, pbsSubtractOriginal
from syserol.tensor import perform_CMTF, tensor_degFreedom, cp_normalize, reorient_factors, sort_factors
from matplotlib.ticker import ScalarFormatter
from itertools import groupby



def makeFigure():
    ax, f = getSetup((13, 9), (1, 4))

    comps = np.arange(1,3)
    tensor, _ = Tensor4D()
    CMTFfacs = [perform_CMTF(tensor, cc) for cc in comps]

    CMTFfacs = [cp_normalize(f) for f in CMTFfacs]
    CMTFfacs = [reorient_factors(f) for f in CMTFfacs]
    CMTFfacs = [sort_factors(f) if i > 0 else f for i,
                f in enumerate(CMTFfacs)]

    Rlabels, agLabels = dimensionLabel3D()
    days = dayLabels()
    tfac = CMTFfacs[1]

    df = pbsSubtractOriginal()
    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components,
              list(df.loc[np.unique(df['patient_ID'])]['group']), "Subjects", ax[0], True)
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[1])
    comp_plot(tfac.factors[2], components, Rlabels, "Receptors", ax[2])
    comp_plot(tfac.factors[3], components, days, "Time (days)", ax[3])
    
    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax, d=False):
    """ Creates heatmap plots for each input dimension by component. """
    if d:
        b = [list(g) for _, g in groupby(ylabel)]
        newLabels = []
        for i, c in enumerate(b):
            newLabels.append([x + "  " if i == len(c)//2 else "–" if i ==
                              0 or i == len(c) - 1 else "·" for (i, x) in enumerate(c)])

        newLabels = [item for sublist in newLabels for item in sublist]

        sns.heatmap(factors, cmap="PiYG", center=0,
                    xticklabels=xlabel, yticklabels=newLabels, ax=ax)
    else:
        sns.heatmap(factors, cmap="PiYG", center=0,
                    xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_title(plotLabel)

