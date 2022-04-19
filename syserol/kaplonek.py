""" Analyze Kaplonek Covid-19 data set in 4D for comparison"""
import pandas as pd
import xarray as xa
from tensordata.kaplonek import importMGH

def kaplonek_4D():
    MGH_data, MGH_subjects, MGH_rec_names, MGH_unique_rec_names, MGH_unique_ant_names = importMGH()

    # Cut off functional data
    if MGH_data[0, :].size != MGH_rec_names.size:
        MGH_data = MGH_data[:, :MGH_rec_names.size]

    sampIDX = pd.MultiIndex.from_tuples(pd.Series(MGH_subjects).str.split("_"), names=["Subjects", "Days"])
    indexx = pd.MultiIndex.from_product([MGH_unique_rec_names, MGH_unique_ant_names], names=['Receptors', 'Antigen'])
    xda = xa.DataArray(MGH_data,
                       dims=["Samples", "Measurement"],
                       coords=[sampIDX, indexx])

    xda = xda.unstack()
    return xda, [list(x) for x in xda.coords.values()]
