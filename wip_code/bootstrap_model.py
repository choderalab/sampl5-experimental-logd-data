from numpy.random import normal, choice
import numpy as np
import pandas as pd
import seaborn as sns


def cyclohexane_dilution_bootstrap(expected_chx_volume=10., expected_oct_volume=90.):
    """
    Bootstrapping model for cyclohexane dilution.

    Parameters
    ----------
    expected_chx_volume - the expected volume of cyclohexane
    expected_oct_volume - the expected volume of octanol that was added in order to dilute cyclohexane

    Returns
    -------

    """
    chx_inaccuracy = 0.1
    chx_imprecision = 0.1
    oct_inaccuracy = 0.05
    oct_imprecision = 0.05

    # We allow the relative bias to be of different scale between the different volumes pipetted,
    # but must be same direction.

    # Draw a bias for the chx pipette
    bias_chx = normal() # draw random normal variate that will scale inaccuracy to determine relative bias
    # draw a bias for the octanol pipette
    bias_oct = normal()
    # Initialize storage for actual concentrations and volumes.

    # Fill initial well of with appropriate dilution of stock solution.
    # TODO do we by definition use same bias for both?
    Vchx_actual = expected_chx_volume * ((1 + chx_inaccuracy * bias_chx) + chx_imprecision * normal())
    Voct_actual = expected_oct_volume * ((1 + oct_inaccuracy * bias_oct) + oct_imprecision * normal())

    return Vchx_actual / (Vchx_actual+Voct_actual)


def measurement_bootstrap(chx_signal, buffer_signal):
    """
    Bootstrapping model for errors in signal integration
    Parameters
    ----------
    signal

    Returns
    -------

    """
    signal_inaccuracy = 0.1
    signal_imprecision = 0.1

    # Bias should be the same for chx and buffer
    # TODO could be wrong if one of the two samples saturates
    bias = normal()

    chx_signal_actual = chx_signal * ((1 + signal_inaccuracy * bias) + signal_imprecision * normal())
    buffer_signal_actual = buffer_signal * ((1 + signal_inaccuracy * bias) + signal_imprecision * normal())
    return chx_signal_actual, buffer_signal_actual


def resample_repeats(measurements):
    n = measurements.shape[0]
    return measurements[choice(n,n)]


def resample_replicates(measurements):
    measurements = resample_repeats(measurements)
    for m, meas in enumerate(measurements):
        n = meas.shape[0]
        measurements[m] = meas[choice(m,m)]
    return measurements


def sample(measurements, resample_measurements=False):
    """

    Parameters
    ----------
    buffer_injection_volume
    chx_injection_volume
    resample_measurements - bool
        Resample real data values
    measurements - numpy array
        All measurements for a single compound
        [
            [repeat 1  [chx_signal, chx_volume, buffer_signal, buffer_volume]_1, ..., [chx_signal, chx_volume, buffer_signal, buffer_volume]_nrepl],
            [repeat 2  [chx_signal, chx_volume, buffer_signal, buffer_volume]_1, ..., [chx_signal, chx_volume, buffer_signal, buffer_volume]_nrepl],
            ...
            [repeat n [chx_signal, chx_volume, buffer_signal, buffer_volume]_1, ..., [chx_signal, chx_volume, buffer_signal, buffer_volume]_nrep]
        ]

    Returns
    -------
    Log D - float
    """

    # Need ndarray for this to work
    assert type(measurements) == np.ndarray

    # Make sure data exists as quartets
    assert measurements.shape[-1] == 4

    if resample_measurements:
        measurements = resample_repeats(measurements)

    result = np.empty(measurements.size / measurements.shape[-1])  # Number of log D estimates/measurements
    for r, repeat in enumerate(measurements):
        # Every replicate will have the same dilution factor
        actual_dilution_factor =  cyclohexane_dilution_bootstrap()
        for i, measurement in enumerate(repeat):
            chx = measurement[0]
            chx_injection_volume = measurement[1]
            buffer = measurement[2]
            buffer_injection_volume = measurement[3]

            # Randomly sample bias and imprecisions in signal
            # Assumption is that the bias in each measurement is independent,
            # because the integration is done independently of the other measurements.
            actual_chx_signal, actual_buffer_signal = measurement_bootstrap(chx,buffer)
            actual_chx_signal /= actual_dilution_factor
            proportional_chx_concentration = (actual_chx_signal / chx_injection_volume)
            proportional_buffer_concentration = (actual_buffer_signal / buffer_injection_volume)
            index = r * len(repeat) + i
            result[index] = np.log10(proportional_chx_concentration / proportional_buffer_concentration)


    return result


# This part is run when this script is executed directly

if __name__ == "__main__":



    # Input data from the preprocessing step.
    table = pd.read_excel('processed.xlsx', sheetname='Filtered Data')

    # Multiply all cyclohexane volumes by 10 to remove implicit dilution factor
    # As a quick fix, they're easily identified in this table as being smaller than 1 microliter),
    table["Volume"] = table["Volume"].apply(lambda x: x * 10.0 if x < 0.99 else x)

    # Store data as convenient arrays in dictionary by compound name
    data_dict = dict()
    # Every individual repeat experiment (completely independent)
    for compound, compound_table in table.groupby("Sample Name"):
        dataset = list()
        for (dset, repeat), repeat_table in compound_table.groupby(["Set", "Repeat"]):
            dataset.append(list())
            for replicate ,replicate_table in repeat_table.groupby("Replicate"):

                solvents = replicate_table.groupby(["Solvent"])
                chx = solvents.get_group('CHX')
                buffer = solvents.get_group('PBS')
                chx_signal = float(chx["Analyte Peak Area (counts)"])
                chx_injection_volume = float(chx["Volume"])
                buffer_signal = float(buffer["Analyte Peak Area (counts)"])
                buffer_injection_volume = float(buffer["Volume"])
                dataset[-1].append([chx_signal, chx_injection_volume, buffer_signal, buffer_injection_volume])

        # Final structure of each compounds array
        # [
        #  [repeat 1 [chx_signal,chx_volume,buffer_signal,buffer_volume]_1, .., [chx_signal,chx_volume,buffer_signal,buffer_volume]_nrepl],
        #  [repeat 2 [chx_signal,chx_volume,buffer_signal,buffer_volume]_1, .., [chx_signal,chx_volume,buffer_signal,buffer_volume]_nrepl],
        #  ...
        #  [repeat n [chx_signal,chx_volume,buffer_signal,buffer_volume]_1, .., [chx_signal,chx_volume,buffer_signal,buffer_volume]_nrepl]
        # ]

        data_dict[compound] = np.asarray(dataset)

    # The good part starts here


    # Number of samples per compound
    n_samples = 100

    for compound in sorted(data_dict.keys()):
        samples = [-10000] * n_samples
        for i in range(n_samples):

            samples[i] = sample(data_dict[compound], resample_measurements=True)

        print(compound, np.average(samples) , np.std(samples))



