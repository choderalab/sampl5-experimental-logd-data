import pandas as pd
import pymc
import math
import numpy
import seaborn as sns
from uncertainties import ufloat
from distributionmodel import LogDModel
import os
from pymbar.timeseries import detectEquilibration, subsampleCorrelatedData
sns.set_style('white')


# http://randlet.com/blog/python-significant-figures-format/
def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)


# Input data from the preprocessing step.
table = pd.read_excel('processed.xlsx', sheetname='Filtered Data')

x = LogDModel(table)

db = pymc.database.txt.load('sampl5_mcmc.txt')
mc = pymc.MCMC(x, db=db)
debug = open("means_vs_median.txt", "w")
out = open("logd_bayes.txt", 'w')
used_samples = open("mcmc_sampling_details.txt","w")
out.write("Molecule, Log D +/-, HPD95%[low, high]\n")
debug.write("Molecule mean - median = difference")
used_samples.write("Molecule, equilibration, N samples")
# curdir = os.getcwd()
# os.makedirs("plots", )
# os.chdir("plots")


for mol in sorted(list(x.logd.keys())):
    print("Processing {}".format(mol))
    # sns.plt.figure()
    trace = numpy.asarray(mc.trace("LogD_{}".format(mol))[:])
    # Burn in and thinning estimated using pymbar
    burnin = detectEquilibration(trace)[0]
    trace= trace[burnin:]
    uncorrelated_indices = subsampleCorrelatedData(trace)
    trace=trace[uncorrelated_indices]

    median = pymc.utils.quantiles(trace)[50]
    mean = numpy.mean(trace)
    lower, upper = pymc.utils.hpd(trace, 0.05)
    lower_s = to_precision(lower,2) # string of number with 2 sig digits
    upper_s = to_precision(upper,2)
    logd = ufloat(mean, numpy.std(trace))

    # Formats the mean and error by the correct amount of significant digits
    out.write("{0}, {1:.1u}, [{2}, {3}]\n".format(mol, logd, lower_s, upper_s ))
    debug.write("{}: {} - {} = {}".format(mol, mean, median, mean-median))
    used_samples.write("{}, {}, {}".format(mol, burnin, len(uncorrelated_indices)))
    # pymc.Matplot.plot(trace, "LogD_{}".format(mol))
    # sns.plt.figure()
# os.chdir(curdir)