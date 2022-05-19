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
import inspect

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
    "c{cterms:02d}_w{wterms:02d}_cf{fit_cterms:02d}_wf{fit_wterms:02d}_smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_antsim-{antsim}",
)
precal.FOLDER = "alan_cal"
precal.DEFAULT_KWARGS['antsim'] = False

get_kwargs = precal.get_kwargs
get_label = precal.get_label
get_likelihood = precal.get_likelihood

class MCMCBoundsError(ValueError):
    pass

cterms = click.option("-c", "--cterms", default=6)
wterms = click.option("-w", "--wterms", default=5)
fit_cterms = click.option("--fit-cterms", default=None, type=int)
fit_wterms = click.option("--fit-wterms", default=None, type=int)
antsim = click.option("--antsim/--no-antsim", default=False)


@main.command()
@cterms
@wterms
@fit_cterms
@fit_wterms
@antsim
@precal.resume
@precal.smooth
@precal.tns_width
@precal.nlive_fac
@precal.optimize
@precal.clobber
@precal.set_widths
@precal.tns_mean_zero
@precal.ignore_sources
@precal.as_sim
@precal.log_level
@precal.s11_sys
@precal.run_mcmc
@precal.opt_iter
@precal.unweighted
@precal.cable_noise_factor
@precal.ndelay
@precal.nscale
def run(**kwargs):
    precal.clirun(**kwargs)


if __name__ == "__main__":
    run()
