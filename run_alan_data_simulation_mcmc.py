import click
import alan_data_utils as utils
import alan_data as adata
from edges_cal.modelling import LinLog
from pathlib import Path
import numpy as np
from p_tqdm import p_map
import run_alan_precal_mcmc as precal
import run_alan_cal_mcmc as cal
import run_alan_data_mcmc as dmc
from functools import partial
main = click.Group()


precal.LABEL_FORMATS = (
    "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_s11{s11_sys}_antsim{antsim}_fg{nterms_fg}_simul{simultaneous}_taufx{fix_tau}_ns{nscale:02d}_nd{ndelay:02d}_noise{add_noise}",
)
precal.FOLDER = "alan_field_and_cal_simulation"
precal.DEFAULT_KWARGS['antsim'] = False

precal.get_likelihood = partial(dmc.get_likelihood, as_sim=('open', 'short', 'hot_load', 'ambient'), sim_sky=True)

@main.command()
@cal.cterms
@cal.wterms
@cal.fit_cterms
@cal.fit_wterms
@cal.antsim
@precal.resume
@precal.smooth
@precal.tns_width
@precal.nlive_fac
@precal.optimize
@precal.clobber
@precal.set_widths
@precal.tns_mean_zero
@precal.ignore_sources
@precal.log_level
@precal.s11_sys
@precal.run_mcmc
@precal.opt_iter
@precal.ndelay
@precal.nscale
@click.option("--add-noise/--no-noise", default=True)
@click.option('--nterms-fg', default=5)
@click.option('--fix-tau/--no-fix-tau', default=True)
@click.option('--simultaneous/--isolated', default=True)
@click.option('--seed', default=1234)
def run(**kwargs):
    precal.clirun(**kwargs)

if __name__ == '__main__':
    run()