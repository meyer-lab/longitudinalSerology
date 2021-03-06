"""Import Zohar data, tensor formation, plotting raw data."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def pbsSubtractOriginal():
    """ Paper Background subtract, will keep all rows for any confusing result. """
    Cov = pd.read_csv("syserol/data/ZoharCovData.csv", index_col=0)
    # 23 (0-> 23) is the start of IgG1_S
    Demographics = Cov.iloc[:, 0:23]
    Serology = Cov.iloc[:, 23::]
    Serology -= Serology.loc["PBS"].values.squeeze()
    df = pd.concat([Demographics, Serology], axis=1)
    df = df.loc[np.isfinite(df["patient_ID"]), :]
    df["week"] = np.array(df["days"] // 7 + 1.0, dtype=int)
    df["patient_ID"] = df["patient_ID"].astype('int32')
    df["group"] = pd.Categorical(df["group"], ["Negative", "Mild", "Moderate", "Severe", "Deceased"])
    df = df.sort_values(by=["group", "days", "patient_ID"])
    df = df.reset_index()
    # Get rid of any data that doesn't have a time component (i.e. "nan" for day)
    df = df.dropna(subset=["days"]).reset_index(drop=True)
    return df


def to_slice(subjects, df):
    Rlabels, AgLabels = dimensionLabel3D()
    tensor = np.full((len(subjects), len(AgLabels), len(Rlabels)), np.nan)
    missing = 0

    for rii, recp in enumerate(Rlabels):
        for aii, anti in enumerate(AgLabels):
            try:
                dfAR = df[recp + "_" + anti]
                dfAR = dfAR.groupby(by="patient").mean()
                dfAR = dfAR.reindex(subjects)
                tensor[:, aii, rii] = dfAR.values
            except KeyError:
                #print(recp + "_" + anti)
                missing += 1

    return tensor

def dayLabels(short=False):
    """ Returns day labels for 4D tensor"""
    df = pbsSubtractOriginal()
    days = np.unique(df["days"])

    if short:
        days = days[days < 16]

    return days


def earlyDaysdf():
    df = pbsSubtractOriginal()
    df = df.loc[df["days"] < 16]
    df.reset_index(drop=True, inplace=True)

    return df

def Tensor4D(df=None):
    """ Create a 4D Tensor (Subject, Antigen, Receptor, Time) """
    if df is None:
        df = pbsSubtractOriginal()
    subj_indexes = np.unique(df['patient_ID'], return_index=True)[1]
    # preserve order of subjects
    subjects = [df['patient_ID'][index] for index in sorted(subj_indexes)]
    Rlabels, AgLabels = dimensionLabel3D()
    days = np.unique(df["days"])
    ndf = df.iloc[:, np.hstack([[1,10], np.arange(23, len(df.columns))])]

    tensor = np.full((len(subjects), len(AgLabels), len(Rlabels), len(days)), np.nan) # 4D

    for i in range(len(ndf)):
        row = ndf.iloc[i, :]
        patient = np.where(row['patient_ID']==subjects)[0][0]
        day = np.where(row['days']==days)[0][0]
        for j in range(2, len(ndf.columns)):
            key = ndf.columns[j].split('_')
            try:
                rii = Rlabels.index(key[0])
                aii = AgLabels.index(key[1])
                tensor[patient, aii, rii, day] = ndf.iloc[i, j]
            except:
                pass

    tensor = np.clip(tensor, 10.0, None)
    tensor = np.log10(tensor)

    # Mean center each measurement
    tensor -= np.nanmean(tensor, axis=0)

    return tensor, np.array(df.index)


def dimensionLabel3D():
    """Returns labels for receptor and antigens, included in the 4D tensor"""
    receptorLabel = [
        "IgG1",
        "IgG2",
        "IgG3",
        "IgA1",
        "IgA2",
        "IgM",
        "FcRalpha",
        "FcR2A",
        "FcR2B",
        "FcR3A",
        "FcR3B"
    ]
    antigenLabel = ["S", "RBD", "N", "S1", "S2", "S1 Trimer"]
    return receptorLabel, antigenLabel


def COVIDpredict(tfac, df=None):
    """ Run Cross-Validated Logistic Regression for COVID Patients"""
    if df is None:
        df = pbsSubtractOriginal()
    patients = np.unique(df['patient_ID'], return_index=True)
    subjj = df.iloc[np.sort(patients[1])]['group'].isin(["Severe", "Deceased"])

    X = tfac.factors[0][subjj, :]
    y = pd.factorize(df.loc[subjj[subjj].index,"group"])[0]
    aucs = []

    kf = StratifiedKFold(n_splits=10, shuffle=True)
    outt = pd.DataFrame({'fold':pd.Series([], dtype='int'),
                   'FPR':pd.Series([], dtype='float'),
                   'TPR':pd.Series([], dtype='float')})
    for ii, (train, test) in enumerate(kf.split(X, y)):
        model = LogisticRegression().fit(X[train], y[train])
        y_score = model.predict_proba(X[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        aucs.append(roc_auc_score(y[test], y_score[:, 1]))
        outt = pd.concat([outt, pd.DataFrame(data={"fold": [ii+1] * len(fpr), "FPR": fpr, "TPR": tpr})])

    xs = pd.unique(outt["FPR"])
    ipl = pd.DataFrame({'fold':pd.Series([], dtype='int'),
                   'FPR':pd.Series([], dtype='float'),
                   'TPR':pd.Series([], dtype='float')})
    for ii in range(kf.n_splits):
        ys = np.interp(xs, outt.loc[outt["fold"]==(ii+1), "FPR"], outt.loc[outt["fold"]==(ii+1), "TPR"])
        ys[0] = 0
        ipl = pd.concat([ipl, pd.DataFrame(data={"fold": [(ii+1)] * len(xs), "FPR": xs, "TPR": ys})])

    return ipl, aucs


def time_components_df(tfac, condition=None):
    subj = pbsSubtractOriginal()
    df = pd.DataFrame(tfac.factors[0])
    comp_names = ["Comp. " + str((i + 1)) for i in range(tfac.factors[0].shape[1])]
    df.columns = comp_names
    df['days'] = subj['days'].values
    df['group'] = subj['group'].values
    df['week'] = subj['week'].values
    if condition is not None:
        df = df.loc[(subj["group"] == condition).values, :]
    df = df.dropna()
    df = pd.melt(df, id_vars=['days', 'group', 'week'], value_vars=comp_names)
    df.rename(columns={'variable': 'Factors'}, inplace=True)
    return df
