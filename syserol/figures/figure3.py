""" This creates Figure 3, for analyzing factorization's ability to predict patient outcome"""
import numpy as np
import seaborn as sns
import pandas as pd
from syserol.COVID import Tensor4D, pbsSubtractOriginal, COVIDpredict, earlyDaysdf
from syserol.tensor import perform_contTF
from .common import getSetup, subplotLabel
from tensorpack import perform_CP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import sem


def makeFigure():
    ax, f = getSetup((9, 6), (3, 2))

    df = pbsSubtractOriginal()
    patients = np.unique(df['patient_ID'], return_index=True)
    patient_type = list(df.iloc[np.sort(patients[1])]['group'])

    # Easy ML, do simple comparison of subject component weights in different groups 
    tFac = perform_contTF()
    group_plot(tFac, patient_type, ax[0])
    tFac_2 = perform_contTF(r=2)
    group_plot(tFac_2, patient_type, ax[1], sev_dec=False, boxplot=True)
    
    # Logistic Regression plots
    roc_df, auc = COVIDpredict(tFac)
    log_plot(roc_df, auc, ax[2])

    # Run regular CP on 4D tensor and do logistic regression comparison
    tensor, _ = Tensor4D()
    CP = perform_CP(tensor, 6)
    roc_CP, auc_CP = COVIDpredict(CP)
    log_plot(roc_CP, auc_CP, ax[3], continuous=False)

    # Predict using only days 0-15
    df = earlyDaysdf()
    tensor, _ = Tensor4D(df)
    tfac_er = perform_contTF(tensor)
    roc_er, auc_er = COVIDpredict(tfac_er, df)
    log_plot(roc_er, auc_er, ax[4])

    subplotLabel(ax)
    return f


def group_plot(tFac, patient_type, ax, sev_dec=True, boxplot=False,):
    components = [str(ii + 1) for ii in range(tFac.rank)]
    subject_df = pd.DataFrame(tFac.factors[0], columns=components)
    subject_df["Patient Status"] = patient_type
    fig_df = pd.melt(subject_df, id_vars=["Patient Status"], var_name="Component", value_name="Component Weight")
    if sev_dec:
        fig_df = fig_df.loc[fig_df["Patient Status"].isin(["Deceased", "Severe"])]
    
    if boxplot:
        sns.boxplot(x="Component", y="Component Weight", hue="Patient Status", data=fig_df, ax=ax)
    else:
        sns.stripplot(x="Component", y="Component Weight", hue="Patient Status", data=fig_df, dodge=True, ax=ax)


def log_plot(roc_df, auc, ax, continuous=True):
    roc_sum = roc_df.groupby(['FPR'], as_index=False).agg({'TPR':['mean','sem']})
    sns.lineplot(x=roc_sum["FPR"], y=roc_sum["TPR"]["mean"], color='b', ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], color="black", ax=ax)
    tprs_upper = np.minimum(roc_sum["TPR"]["mean"] + roc_sum["TPR"]["sem"], 1)
    tprs_lower = np.maximum(roc_sum["TPR"]["mean"] - roc_sum["TPR"]["sem"], 0)
    ax.fill_between(roc_sum["FPR"], tprs_lower, tprs_upper, color='grey', alpha=.2)
    if continuous:
        ax.set_title("Continuous Factorization - Severe vs. Deceased ROC (AUC={}±{})".format(np.around(np.mean(auc), decimals=2),
                                                                    np.around(sem(auc), decimals=2)))
    else:
        ax.set_title("4D Regular CP - Severe vs. Deceased ROC (AUC={}±{})".format(np.around(np.mean(auc), decimals=2),
                                                                    np.around(sem(auc), decimals=2)))
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")


