import matplotlib
matplotlib.rc('text', usetex=True)
from matplotlib import pyplot as plt

matplotlib.rc('font', size='22')
#http://stackoverflow.com/a/20709149
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',
    r'\sisetup{detect-all}',
    r'\usepackage{helvet}',
    r'\usepackage{sansmath}',
    r'\sansmath'
]
import pandas as pd
import seaborn as sns
from copy import deepcopy
sns.set(style="white")
import numpy as np
from math import log10, floor
import sys
from uncertainties import ufloat
from collections import OrderedDict
from openeye.oechem import *
from openeye.oedepict import *



avg_data = pd.read_csv("logD_sem.txt")
sqrt3_data = pd.read_csv("logD_sqrt3.txt")
bootstrap_nonparm = pd.read_csv("LogD_nonparametric_bootstrap.txt")
bootstrap_parm = pd.read_csv("LogD_parametric2_bootstrap.txt")
sqrt3_data["Method"] = "Sqrt3"
avg_data["Method"] = "Average"


# print(bootstrap_parm.to_latex())
bootstrap_nonparm["Method"] = "Non-parametric bootstrap"
bootstrap_parm['Method'] = "Parametric bootstrap"

logD_data = pd.concat([avg_data, sqrt3_data, bootstrap_nonparm, bootstrap_parm])
# log D +/- uncertainty
logD_data["xpmy"] = logD_data.apply(lambda row: "{:.1u}".format(ufloat(row.logD,row.uncertainty)), axis=1)


param = logD_data.query("Method == 'Parametric bootstrap'")
avg = logD_data.query("Method == 'Average'")
sqrt3 = logD_data.query("Method == 'Sqrt3'")
nonparam = logD_data.query("Method == 'Non-parametric bootstrap'")

data_per_compound = open('compounddata.tex', 'w')
tablecopy = logD_data.pivot("Compound", "Method")
tablecopy.drop('logD', axis=1, inplace=True)
tablecopy.drop('uncertainty', axis=1, inplace=True)

tablecopy.to_latex(buf=data_per_compound, index=True)


def hacked_statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using statsmodels. Modified to give the exponentiated values on the y axis"""
    import statsmodels.nonparametric.api as smnp
    from seaborn.utils import _kde_support
    if isinstance(bw, str):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, clip[0])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # exponentiate y values
    return xx, np.exp(yy), z


sns.distributions._statsmodels_bivariate_kde = hacked_statsmodels_bivariate_kde

def plot2dkdeforme(dataset, cmap, scatterplus=True, title='', xlabel=r'$\log \mathrm{D}$', ylabel=r'Standard error'):
    fig = plt.figure()
    ax = sns.kdeplot(dataset.logD, np.log(dataset.uncertainty),
                     cmap=cmap, shade=True, n_levels=9, shade_lowest=False, alpha=0.88, bw=[0.4, 0.3], kernel='gau', cbar=True)
    if scatterplus:
        scat = plt.scatter(dataset.logD, dataset.uncertainty, c='w', marker='o', s=9, edgecolors='k', linewidths=0.9)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22)

    plt.title(title, fontsize=22)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.xlim(-5,4)
    plt.ylim(0.0,1.4)
    plt.tight_layout()

    return ax

plot2dkdeforme(avg, cmap='Greys', title='Average')
plt.savefig("average.pdf", dpi=150)
plot2dkdeforme(sqrt3, cmap='Greys', title='Average',  ylabel=r'$\sigma\sqrt{3}$')
plt.savefig("sqrt3.pdf", dpi=150)
plot2dkdeforme(param, cmap='Greys', title='Parametric bootstrap')
plt.savefig("parametric.pdf", dpi=150)
plot2dkdeforme(nonparam, cmap='Greys', title='Nonparametric bootstrap')
plt.savefig("nonparametric.pdf", dpi=150)

print(logD_data.groupby("Method").describe())

plt.show()

