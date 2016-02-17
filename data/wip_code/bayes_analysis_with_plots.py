import pandas as pd
import pymc
import math
import numpy
import seaborn as sns
from uncertainties import ufloat
from moremodels import DispensingErrorModel
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

x = DispensingErrorModel(table)

db = pymc.database.txt.load('sampl5_test.txt')
mc = pymc.MCMC(x, db=db)
out = open("logd_bayes_test.txt", 'w')
out.write("Molecule, Log D +/-, HPD95%[low, high]\n")
curdir = os.getcwd()

# os.makedirs("testplots", )
os.chdir("testplots")


mols = numpy.unique(table["Sample Name"])

# for mol in sorted(list(x.model.keys())):
#     if mol[-1] in "ABC": continue
# for mol in sorted(mols):
for param in sorted(list(x.model.keys())):
    if param[0] == "C":
        # variables starting with C are our observations and dont have traces
        continue
    print("Processing {}".format(param))
    sns.plt.figure()
    trace = numpy.asarray(mc.trace(param)[:])
    # Burn in and thinning estimated using pymbar
    # trace2=trace[detectEquilibration(trace)[0]:]
    # trace2=trace2[subsampleCorrelatedData(trace2)]

    median = pymc.utils.quantiles(trace)[50]
    lower, upper = pymc.utils.hpd(trace, 0.05)
    lower_s = to_precision(lower,2) # string of number with 2 sig digits
    upper_s = to_precision(upper,2)
    logd = ufloat(median, numpy.std(trace))

    # Formats the mean and error by the correct amount of significant digits
    out.write("{0}, {1:.5u}, [{2}, {3}]\n".format(param, logd, lower_s, upper_s ))

    pymc.Matplot.plot(trace, param)
    # sns.plt.figure()
    # pymc.Matplot.plot(trace2, "thinned_LogD_{}".format(mol))
os.chdir(curdir)