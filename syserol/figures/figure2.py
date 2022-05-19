import numpy as np
import seaborn as sns
import pandas as pd
from .common import getSetup, subplotLabel
from syserol.COVID import Tensor4D, dayLabels, dimensionLabel3D, pbsSubtractOriginal, earlyDaysdf
from syserol.tensor import perform_contTF
from itertools import groupby



def makeFigure(tensor=None):
    ax, f = getSetup((6, 7), (2, 2))

    if tensor is None:
        df = earlyDaysdf()
        tensor, _ = Tensor4D(df)

    Rlabels, agLabels = dimensionLabel3D()
    days = dayLabels(short=True)
    tfac = perform_contTF(tensor, 5)

    #df = pbsSubtractOriginal()
    components = [str(ii + 1) for ii in range(tfac.rank)]
    patients = np.unique(df['patient_ID'], return_index=True)
    comp_plot(tfac.factors[0], components,
              list(df.iloc[np.sort(patients[1])]['group']), "Subjects", ax[0], True)
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[1])
    comp_plot(tfac.factors[2], components, Rlabels, "Receptors", ax[2])
    lineplot(tfac, days.astype(int), "Time (days)", ax[3])
    
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

        sns.heatmap(factors, cmap="PiYG",
                    xticklabels=xlabel, yticklabels=newLabels, vmin=-1, vmax=1, ax=ax)
        
    else:
        sns.heatmap(factors, cmap="PiYG",
                    xticklabels=xlabel, yticklabels=ylabel, vmin=-1, vmax=1, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.set_xlabel("Components", fontsize=11.5)
    ax.set_title(plotLabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)



def lineplot(tfac, days, xlabel, ax):
    components = ["Component " + str(ii + 1) for ii in range(tfac.rank)]
    cont_df = pd.DataFrame(tfac.factors[3], columns=components)
    cont_df[xlabel] = days
    cont_df = cont_df.set_index(xlabel)
    sns.lineplot(data = cont_df, palette ="colorblind", dashes=False, ax=ax)
    ax.set_ylabel("Component Weight", fontsize=11.5)
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.legend(fontsize='medium')
