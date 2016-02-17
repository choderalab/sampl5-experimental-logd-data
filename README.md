# SAMPL5 Distribution coefficients

### Authors

* [Bas Rustenburg](https://github.com/bas-rustenburg)
  * Genentech, South San Francisco, CA,
  * Chodera Lab at Memorial Sloan Kettering, New York, NY
* [John Chodera](http://www.choderalab.org/)
  * Memorial Sloan Kettering, New York, NY
* [David Mobley](http://www.mobleylab.org/)
  * University of California, Irvine, Irvine, CA

## Description

As part of the [SAMPL5 challenge](https://drugdesigndata.org/about/sampl5), cyclohexane-water distribution coefficients (Log D) have been measured at Genentech for 53 small, drug-like molecules. The technique used is a mass spectrometry-based assay as described in [Lin & Pease, 2013](http://www.ncbi.nlm.nih.gov/pubmed/24168238).

Inside this repository you will find raw and processed data files containing the experimental data that was collected, the [final results](./data/logD_final.txt) as well as the python scripts used to analyze the data.


## Files

See [logD_final.txt](./logD_final.txt) for final data values, generated by [`analysis.py`](./analysis.py).

The experimental data that was used for the analysis is found in [`processed.xlsx`](./processed.xlsx).
    
Additionally, unprocessed, raw data files are found in [`raw_data`](./raw_data).

See `docs/index.html` for a formatted documentation of the analysis.

## Acknowledgements 
* Dan Ortwine, Genentech
* Baiwei Lin, Genentech
* Joseph Pease, Genentech
* Justin Dancer, Genentech
* JW Feng, Denali Therapeutics
* Caitlin Bannan, University of California, Irvine // Mobley Lab



