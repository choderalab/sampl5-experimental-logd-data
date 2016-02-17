#!/usr/bin/env python3
# Analyzes the data with propagation of uncertainties by linear approximation error propagation theory as implemented in uncertainties package.
# This script only works for python 3.

# Depends on pandas and uncertainties package.
# Uncertainties: a Python package for calculations with uncertainties, Eric O. LEBIGOT, http://pythonhosted.org/uncertainties/
import pandas as pd
from uncertainties import ufloat
from uncertainties.umath import log10
import math

# Input data from the preprocessing step.
table = pd.read_excel('processed.xlsx', sheetname='Filtered Data')

# Final output file
output = open('logD_final.txt', 'w')

# Store results in new dataframe
replicate_measurements = pd.DataFrame()

# Estimate the uncertainty from technical replicates of the same experiments.
# This is done by averaging over multiple injections of the same solution and calculating the standard deviation.
for (compound, dset, repeat, solv), df_exp in table.groupby(["Sample Name", "Set", "Repeat", "Solvent", ]):
    mean = df_exp["Area/Volume"].mean()
    # Multiply by sqrt(3) since replicate measurements not independent
    uncertainty = df_exp["Area/Volume"].std() * math.sqrt(3.0)

    # Store the log base 10 of the measurement.
    measurement = log10(ufloat(mean, uncertainty, tag="replicate"))
    replicate_measurements = replicate_measurements.append(
            {"Compound": compound, "Solvent": solv, "Log10 Area/Volume (Uncertainty)": measurement,
             "Experiment": "{}-{}".format(dset, repeat)}, ignore_index=True)

# Log D estimates are tabulated in a new dataframe.
experiments = pd.DataFrame()

# Calculate the log D of each individual repeat experiment
for (compound, measurement), estimates in replicate_measurements.groupby(["Compound", "Experiment"]):
    groups = estimates.groupby(["Solvent"])
    chx = groups.get_group('CHX')
    pbs = groups.get_group('PBS')

    # Error checking, these should be single experiment values, or the following hack won't work correctly.
    assert len(chx) == 1
    assert len(pbs) == 1

    # Doing math on the pd.Series of length one returns Nans as uncertainty. This hack extracts the value.
    log_chx = chx["Log10 Area/Volume (Uncertainty)"]._values[0]
    log_pbs = pbs["Log10 Area/Volume (Uncertainty)"]._values[0]

    # log D is defined as  log_10 (peak area/injection volume CHX) - log_10 (peak area/injection volume PBS),
    estimate = log_chx - log_pbs
    experiments = experiments.append({"Compound": compound, "log D": estimate}, ignore_index=True)

# Header for the output file
output.write("Compound,log D,uncertainty\n")

# Combine all independent repeat experiments into one estimate of the log D with uncertainty
for compound, group in experiments.groupby("Compound"):
    tot = 0.0
    for val in group["log D"]:
        tot += val

    tot /= len(group)

    # The final estimate is written to the output file
    # Rounds uncertainty to 1 sig digit, and rounds value appropriately based on uncertainty.
    output.write("{}, {:.1u}\n".format(compound,tot))

output.close()



