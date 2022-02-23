""" Analyze Kaplonek Covid-19 data set in 4D for comparison"""
import numpy as np
from tensordata.kaplonek import importMGH, load_file

def kaplonek_4D():
    MGH_data, _ , MGH_rec_names, MGH_unique_rec_names, MGH_unique_ant_names = importMGH()

    # Cut off functional data
    if MGH_data[0, :].size != MGH_rec_names.size:
        MGH_data = MGH_data[:, :MGH_rec_names.size]

    # Create an array of the column indices corresponding to each unique receptor
    rec_ind = np.zeros((MGH_unique_rec_names.size, int(MGH_rec_names.size / MGH_unique_rec_names.size))).astype(int)

    for xx in range(MGH_unique_rec_names.size):
        rec_index = np.where(MGH_rec_names == MGH_unique_rec_names[xx])
        rec_index = np.array(rec_index)
        rec_ind[xx, :] = rec_index

    # Load in additional necessary subject/hospitalization day alignment data
    visualize_df = load_file("MGH_Visualizing.Plate.WHO124")
    hosp_days = visualize_df["TimePoints"]
    unique_days = np.unique(hosp_days)
    meta_data = load_file("MGH_Sero.Meta.data.WHO124")
    subject_order = meta_data["Study_ID"]
    unique_subjects = np.unique(meta_data["Study_ID"])
    days_dict = dict({"D0":0, "D3":1, "D7":2})

    # Create 4D cube
    MGH_cube = np.zeros((unique_subjects.size, MGH_unique_rec_names.size, rec_ind[0, :].size, unique_days.size))

    # Fill the cube
    for subj_ind, day in enumerate(hosp_days):
        cur_subj = subject_order[subj_ind]
        cube_ind = np.where(cur_subj == unique_subjects)[0][0]
        for receptor_ind in range(np.size(MGH_cube, 1)):
            day_ind = days_dict[day]
            MGH_cube[cube_ind, receptor_ind, :, day_ind] = MGH_data[subj_ind, rec_ind[receptor_ind, :]] 

    # Below is extra code for checking cube
    # Create a dictionary that matches the unique subject index to the overall indices in the original array
    dicts = {}
    for subj_ind, day in enumerate(hosp_days):
        cur_subj = subject_order[subj_ind]
        dicts[subj_ind] = np.where(cur_subj == unique_subjects)[0][0]

    # Check that cube is filled correctly
    for keys, values in dicts.items():
        day = days_dict[hosp_days[keys]]
        assert (MGH_cube[values, :, :, day].flatten() == MGH_data[keys, :]).all()

    # Collect our new axis labels
    axes = unique_subjects, MGH_unique_rec_names, MGH_unique_ant_names, unique_days

    return MGH_cube, axes
