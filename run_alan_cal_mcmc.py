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

@main.command()
@click.option("-c", "--cterms", default=6)
@click.option("-w", "--wterms", default=5)
@click.option("--fit-cterms", default=6)
@click.option("--fit-wterms", default=5)
@click.option("--resume/--no-resume", default=False)
@click.option("-s", "--smooth", default=1)
@click.option("-p", "--tns-width", default=3)
@click.option("-n", "--nlive-fac", default=100)
@click.option("-o", "--optimize", type=click.Choice(['none', 'dual_annealing', 'basinhopping'], case_sensitive=False), default='basinhopping')
@click.option("--clobber/--no-clobber", default=False)
@click.option("--set-widths/--no-set-widths", default=False)
@click.option("--tns-mean-zero/--est-tns", default=True)
@click.option("--antsim/--no-antsim", default=False)
@click.option('--ignore', multiple=True, type=click.Choice(['short', 'open','hot_load', 'ambient', 'AntSim1']))
@click.option('--as-sim', multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient', 'AntSim1']))
@click.option("--log-level", default='info', type=click.Choice(['info', 'debug', 'warn', 'error']))
@click.option("--s11-sys", multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient', 'rcv', 'AntSim1']))
@click.option("--run-mcmc/--no-mcmc", default=True)
@click.option("--opt-iter", default=10)
@click.option("--unweighted/--weighted", default=False)
@click.option("--cable-noise-factor", default=1, type=int)
@click.option("--ndelay", default=1, type=int)
@click.option("--nscale", default=1, type=int)
def run(
    **kwargs
):

    # lc = locals()
    # sig = inspect.signature(run)
    # kwargs = {k: lc[k] for k in sig}

    precal.clirun(**kwargs)


if __name__ == "__main__":
    run()
