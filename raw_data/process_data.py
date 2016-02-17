# Unify the data format from the integrated raw data
# Generates excel sheets with necessary metadata such as experiments, replicate numbers, repeat numbers.

# Depends on pandas, numpy, pyyaml
import pandas as pd
import numpy as np
import yaml


# The data is read from tables containing the integrated peak areas, and metadata, some of which is embedded in the filenames.
# Each file represents a batch of experiments. Each set represents one set of 32 molecules.
# Batch 1 was measured twice, using different injection volumes.

data = dict(b1s1_38=pd.read_table("150917-Batch1Set1-wk38-BAS.txt"), b1s2_39=pd.read_table("150923-Batch1Set2-wk39-BAS.txt"),
            b1s1_39=pd.read_table("150924-Batch1Set1-wk39-BAS.txt"), std_39=pd.read_table("150925-STD-wk39-BAS.txt", header=1))

# Process each dataframe, adding metadata in new columns
for setname, df in data.items():
    df["Solvent"] = df["Sample ID"].apply(lambda x: x[0:3])
    df["Set"] = setname

    # Extracing all of the useful metadata that is contained in the file name.

    # Repeat indicates the independent experiments that was repeated
    # Replicate indicates a technical replicate (multiple injections/measurements of the same solution)

    df["Date"] = df["File Name"].apply(lambda x: x.split(sep='- ')[0][:])
    df["Repeat"] = df["File Name"].apply(lambda x: int(x.split(sep='-')[4][:]))
    df["Replicate"] = df["File Name"].apply(lambda x: x.split(sep='-')[5][:])

    # Store information on the number of microliters injected into LCMS, These are known from the experimental protocol

    # These valumes are already corrected for octanol dilution (10% cyclohexane, 90% octanol).
    # I have no available estimate of the introduced uncertainty there.

    if setname == "b1s1_39":
        # Changed the MS injection volumes for this experiment only.
        # This was done as a means to increase the detection sensitivity in the cyclohexane phase, while reducing possible upper detection
        # limit issues in the PBS phase measurements.
        vols = dict(CHX=0.2, PBS=1)
    else:
        vols = dict(CHX=0.1, PBS=2)

    df["Volume"] = df["Solvent"].apply(lambda x: vols[x])

    # Normalize peak area by volume. Assuming zero uncertainty in injection volumes.
    df["Area/Volume"] = df["Analyte Peak Area (counts)"] / df["Volume"]

    # Precalculating this value for the quick and dirty estimate only.
    df["log10 (Area/Volume)"] = np.log10(df["Area/Volume"])

    # Drop columns that don't contain information (NA columns)
    df = df.drop('Sample Type', 1)
    df = df.drop('Unnamed: 0', 1)
    df = df.drop('Calculated Concentration (ng/mL)', 1)
    data[setname] = df.dropna(axis=1, how='all')

# Merge all data sets.
data = pd.concat(data)

# Create new field in the data table to mark data points that are to be excluded from the dataset in the final analysis,
# due to poor reproducibility.
data["Exclude"] = False

# Loading the list of compounds to exclude from each set of experiments.
exclusions = yaml.load(open('excluded_samples.txt'))

# Data was nit-picked based on reproducibility of experiment and detection limits.

b1s1_38_exc=exclusions["150917-Batch1Set1-wk38-BAS.txt"]
b1s2_39_exc=exclusions["150923-Batch1Set2-wk39-BAS.txt"]
b1s1_39_exc=exclusions["150924-Batch1Set1-wk39-BAS.txt"]
std_39_exc=exclusions["150925-STD-wk39-BAS.txt"]


# Mark compounds that need to be excluded from an individual set.

data.loc[(data["Set"] == "b1s1_38") & (
    data["Sample Name"].isin(b1s1_38_exc)), "Exclude"] = True
data.loc[(data["Set"] == "b1s1_39") & (
    data["Sample Name"].isin(b1s1_39_exc)), "Exclude"] = True
data.loc[(data["Set"] == "b1s2_39") & (
    data["Sample Name"].isin(b1s2_39_exc)), "Exclude"] = True
data.loc[(data["Set"] == "std_39") & (
    data["Sample Name"].isin(std_39_exc)), "Exclude"] = True

# Store all the processed data in an excel sheet

xlsx = pd.ExcelWriter('processed.xlsx')
data_filtered = data[data["Exclude"] == False]
data_filtered.to_excel(xlsx, sheet_name='Filtered Data')
data.to_excel(xlsx, sheet_name='All Data')

# Quick and dirty logD calculation without uncertainties

output = open("logd.txt", "w")
output.write("Compound\tDate\tRepeat\tReplicate\tlog_D\tlog_chx\tlog_pbs\n")
for compound, df_compound in data_filtered.groupby("Sample Name"):
    for (date, repeat, repl), df_exp in df_compound.groupby(["Date", "Repeat", "Replicate",]):
        groups = df_exp.groupby(["Solvent"])
        chx = groups.get_group('CHX')
        pbs = groups.get_group('PBS')
        chx_av = float(chx["log10 (Area/Volume)"])
        pbs_av = float(pbs["log10 (Area/Volume)"])
        output.write("%s\t%s\t%s\t%s\t%.2f\t%.2f\t%.2f\n" % (compound, date, repeat, repl, chx_av - pbs_av, chx_av, pbs_av))

output.close()
xlsx.save()
xlsx.close()