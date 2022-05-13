"""Utilities for doing the simulation exploration."""

from edges_cal import CalibrationObservation
from edges_analysis.analysis.calibrate import LabCalibration
import numpy as np
from edges_estimate.eor_models import AbsorptionProfile
from edges_estimate.likelihoods import DataCalibrationLikelihood, NoiseWaveLikelihood
from edges_cal.modelling import LinLog, Polynomial, UnitTransform
from scipy import stats
from yabf import ParamVec
from pathlib import Path
from edges_cal.simulate import simulate_qant_from_calobs
import matplotlib.pyplot as plt

from alan_data_utils import (
    get_calobs,
    get_labcal,
    calobs as fid_calobs,
    labcal as fid_labcal,
    fid_eor,
    get_tns_params,
)
# The following value is in line with the constant variance used for Alan's data
fid_data_var = 8.4e-11
sky_freqs = np.linspace(53, 99, 120)

fid_fg = LinLog(n_terms=5, parameters=[2000, 10, -10, 5, -5])

def get_tns_model(calobs, tns_width=100, est_tns=None):
    t_ns_params = get_tns_params(calobs, tns_width=tns_width, est_tns=est_tns)
    t_ns_model = Polynomial(parameters=[p.fiducial for p in t_ns_params], transform=UnitTransform(range=(calobs.freq.min, calobs.freq.max)))
    return t_ns_model, t_ns_params

def sim_antenna_q(labcal, fg=fid_fg, eor=fid_eor):
    calobs = labcal.calobs
    
    spec = fg(x=eor.freqs) + eor()['eor_spectrum']
    
    tns_model, _ = get_tns_model(calobs)
    scale_model = tns_model.with_params(tns_model.parameters/calobs.t_load_ns)
        
    ant_s11 = labcal.antenna_s11_model(eor.freqs)

    return simulate_qant_from_calobs(
        calobs, ant_s11=ant_s11, ant_temp=spec, 
        scale_model=scale_model, freq=eor.freqs
    )

def get_cal_likelihood(
    calobs, 
    tns_width: float=3.0, variance='data', seed=None, 
    **kwargs
):
    if seed:
        np.random.seed(seed)
        
    return NoiseWaveLikelihood.from_sim_calobs(
        calobs, 
        variance=variance,
        t_ns_width = tns_width,
        **kwargs
    )

def get_data_likelihood(labcal, qvar_ant=fid_data_var, fg=fid_fg, eor=fid_eor, cal_noise='data', simulate=True, seed=None):
    calobs = labcal.calobs
    
    q = sim_antenna_q(labcal, fg=fg, eor=eor)
    
    if isinstance(qvar_ant, (int, float)):
        qvar_ant = qvar_ant * np.ones_like(eor.freqs)
    
    if seed:
        np.random.seed(seed)
    
    q = q + np.random.normal(scale=qvar_ant)
    
    tns_model, tns_params = get_tns_model(calobs)

    return DataCalibrationLikelihood.from_labcal(
        labcal, 
        q_ant=q, 
        qvar_ant=qvar_ant, 
        fg_model=fg, 
        eor_components=(eor,),
        sim=simulate,
        t_ns_params=tns_params,
        cal_noise=cal_noise,
        field_freq=eor.freqs
    )

def view_results(lk, res_data,  calobs=fid_calobs, label=None, fig=None, ax=None, c=0, eor=fid_eor):
    """Simple function to create a plot of input vs expected TNS and T21."""
    eorspec = lk.partial_linear_model.get_ctx(params=res_data.x)

    if fig is None:
        plot_input = True
        fig, ax = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
    else:
        plot_input = False

    color = f"C{c}"
    
    nu = eor.freqs

    tns_model, _ = get_tns_model(calobs)
    tns_model = tns_model(nu)
    
    if plot_input:
        ax[0, 0].plot(nu, tns_model, label='Input', color='k')
    
    ax[0, 0].plot(nu, eorspec['tns_field'], label='Estimated' + (' '+label if label else ''), color=color)
    
    ax[1, 0].plot(nu,eorspec['tns_field'] - tns_model, label=r"$\Delta T_{\rm NS}$" if plot_input else None, color=color)
    ax[1, 0].plot(nu,(eorspec['tns_field'] - tns_model)*lk.data['q']['ant'], ls='--', color=color, label=r"$\Delta T_{\rm NS} Q_{\rm ant}$" if plot_input else None)
    
    ax[0, 0].set_title(r"$T_{\rm NS}$")
    ax[0, 0].set_ylabel("Temperature [K]")
    
    if plot_input:
        ax[0, 1].plot(nu,eor()['eor_spectrum'], color='k')
    
    ax[0, 1].plot(nu,eorspec['eor_spectrum'])
    ax[0, 1].set_title(r"$T_{21}$")
    delta = eorspec['eor_spectrum'] - eor()['eor_spectrum']
    ax[1, 1].plot(nu, delta, color=color, label=f"Max $\Delta = {np.max(np.abs(delta))*1000:1.2e}$mK")
    ax[1, 0].set_ylabel("Difference [K]")
            
    ax[1, 0].set_xlabel("Frequency")
    ax[1, 1].set_xlabel("Frequency")
    
    ax[0, 0].legend()
    ax[1, 0].legend()
    ax[1,1].legend()
    
    return fig, ax

