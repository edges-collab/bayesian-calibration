# Bayesian Calibration

This folder contains working for the Bayesian Calibration paper(s), and
general development of the idea of Bayesian Calibration in EDGES.

Paper URL: https://www.overleaf.com/9477723471yvqbvyshqjph

The main development notebook is `devel-bayesian-cal.ipynb`: this is a non-production 
notebook where stuff gets tinkered with. Notably, it contains math etc. for the 
variance model. All configuration files, outputs etc. for this notebook are in the
`development/` folder.

Over time, I've tried a few different bits of data and tests etc. First, I tried 
(re-)calibrating Raul's data. His data is in `raul-data/`. I didn't pursue this.

The final results will be with Alan's data (in order to make it as close to the NP as 
possible). All data *coming from Alan* is in `alan-data/`. 

With Alan's data + calibration choices, I do the following:

    0. Data Investigation -- `raw_data_assumptions.ipynb`
        This is done without really requiring calibration or anything. It just looks at
        spectra over time/frequency and evaluates whether our likelihood model 
        assumptions (eg. gaussianity, independence of frequencies) are correct.
    1. Pure Calibration -- `alan_calibration.ipynb`
        a) Simulated Data: the point here is to check whether the relevant likelihoods
           work properly (eg. comparing evidence for correct model versus one with
           unnecessary parameters). The MCMC runner script is `run_cal_simulation_mcmc.py`.
           Outputs in `sim_cal/`.
        b) Real Data: the point here is to figure out what number of terms are required
           to successfully characterize the calibration solutions. Overall, outputs 
           consist of multiple runs against the same data with different numbers of terms.
           MCMC runner script at `run_alan_cal_mcmc.py`.  Outputs in `alan_cal/`
    2. Calibration + Field Data
        a) Simulated Data: the point here goes beyond that things just "work", to 
           exploring the impact of correlations between foregrounds and calibration, eg.
           by introducing more foreground terms in the *data* but not the model. 
           Notebooks are where??
        b) Real Data: the final point of the paper. Produce MCMCs with `run_alan_data_mcmc.py`
           and analyse with `alan-data-calibration.ipynb`

Along with the above scripts/outputs, there is a CLI utils script that helps to deal 
with the MCMC outputs: `mcmc_utils.py` that can be used to eg. print out all runs, 
show evidences, rename things, etc.