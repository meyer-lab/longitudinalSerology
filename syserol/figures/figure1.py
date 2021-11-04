import numpy as np
from .common import getSetup, subplotLabel

def makeFigure():
    ax, f = getSetup((13, 9), (3, 4))
    comps = np.arange(1, 7)
    return f