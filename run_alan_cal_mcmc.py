from __future__ import annotations
import click
import time
import alan_data_utils as utils
import parse
import numpy as np
from rich.logging import RichHandler
from yabf.core import mpi
import logging
from rich.console import Console
from rich.rule import Rule
from typing import Any
from functools import partial
import run_alan_precal_mcmc as precal

cns = Console(width=200)
log = logging.getLogger(__name__)
log.addHandler(RichHandler(rich_tracebacks=True, console=cns))

main = click.Group()

precal.LABEL_FORMATS = (
    "c{cterms:d}_w{wterms:d}_smooth{smooth:d}",
    "c{cterms:d}_w{wterms:d}_smooth{smooth:d}_tns{tns_width:d}",
    "c{cterms:d}_w{wterms:d}_smooth{smooth:d}_tns{tns_width:d}_ign[{ignore_sources}]",
    "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]",
    "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]",
    "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}",
    "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_antsim-{antsim}",
    "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_antsim-{antsim}_s11unif-{s11_unif_prior}",
)
precal.FOLDER = "alan_cal"
precal.DEFAULT_KWARGS['antsim'] = False

get_kwargs = precal.get_kwargs
get_label = precal.get_label

def get_likelihood(cterms, wterms, smooth, tns_width, est_tns=None, ignore_sources=(), as_sim=(), s11_sys=(), antsim=False, s11_unif_prior=False):
    s11_systematic_params = precal.define_s11_systematics(s11_sys, unif=s11_unif_prior)
    
    calobs = utils.get_calobs(cterms=cterms, wterms=wterms, smooth=smooth)

    return utils.get_cal_lk(calobs, tns_width=tns_width, est_tns=est_tns, ignore_sources=ignore_sources, as_sim=as_sim, s11_systematic_params=s11_systematic_params, include_antsim=antsim)

precal.get_likelihood = get_likelihood

class MCMCBoundsError(ValueError):
    pass

@main.command()
@click.option("-c", "--cterms", default=(6,), multiple=True)
@click.option("-w", "--wterms", default=(5,), multiple=True)
@click.option("-l", "--label", default=None, type=str)
@click.option("--resume/--no-resume", default=False)
@click.option("-s", "--smooth", default=1)
@click.option("-p", "--tns-width", default=3)
@click.option("-n", "--nlive-fac", default=100)
@click.option("-o/-O", "--optimize/--no-optimize", default=True)
@click.option("--clobber/--no-clobber", default=False)
@click.option("--set-widths/--no-set-widths", default=False)
@click.option("--tns-mean-zero/--est-tns", default=True)
@click.option("--antsim/--no-antsim", default=False)
@click.option('--ignore', multiple=True, type=click.Choice(['short', 'open','hot_load', 'ambient', 'AntSim1']))
@click.option('--as-sim', multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient', 'AntSim1']))
@click.option("--log-level", default='info', type=click.Choice(['info', 'debug', 'warn', 'error']))
@click.option("--s11-model", multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient', 'rcv', 'AntSim1']))
@click.option("--s11-uniform/--s11-lognorm", default=False)
def run(
    cterms,
    wterms,
    label,
    resume,
    smooth,
    tns_width,
    nlive_fac,
    clobber,
    optimize,
    set_widths,
    tns_mean_zero,
    antsim,
    ignore,
    as_sim,
    log_level,
    s11_model,
    s11_uniform,
):

    root_logger = logging.getLogger('yabf')
    root_logger.setLevel(log_level.upper())
    root_logger.addHandler(RichHandler(rich_tracebacks=True, console=cns))
    
    for c in cterms:
        for w in wterms:
            if mpi.am_single_or_primary_process:
                cns.print(Rule(f'Doing cterms={c} wterms={w}'))
            precal.run_single(
                label,
                resume,
                label_kwargs = {
                    'smooth': smooth,
                    'tns_width': tns_width,
                    'ignore_sources': ignore,
                    'as_sim': as_sim,
                    's11_sys': tuple(s11_model),
                    'cterms': c,
                    'wterms': w,
                    'antsim': antsim,
                    's11_unif_prior': s11_uniform
                },
                nlive_fac=nlive_fac,
                clobber=clobber,
                optimize=optimize,
                set_widths=set_widths,
                est_tns=np.zeros(cterms) if tns_mean_zero else None,
            )


if __name__ == "__main__":
    run()
