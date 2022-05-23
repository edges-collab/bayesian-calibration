import click
from typing import Any
import run_alan_cal_mcmc as cal

import run_alan_precal_mcmc as precal
import run_mcmc_utils as run
from functools import partial


mcdef = run.MCDef(
    label_format = (
        "c{cterms:d}_w{wterms:d}_cf{fit_cterms:d}_wf{fit_wterms:d}_smooth{smooth:d}_tns{tns_width:d}_var-{variance}_s11{s11_sys}_nscale{nscale:02d}_ndelay{ndelay:02d}_noise{add_noise}",
    ),
    folder = "alan_cal_simulation",
    default_kwargs = cal.default_kwargs,
    get_likelihood = partial(cal.precal.get_likelihood, as_sim=('hot_load', 'ambient', 'short', 'open'))
)


main = click.Group()

@main.command()
@run.all_mc_options

@cal.cterms
@cal.wterms
@cal.fit_cterms
@cal.fit_wterms
@precal.smooth
@precal.tns_width
@precal.tns_mean_zero
@precal.ignore_sources
@precal.s11_sys
@precal.ndelay
@precal.nscale
@click.option("--seed", default=1234)
@click.option("--variance", default='data')
@click.option("--add-noise/--no-noise", default=True)
def clirun(**kwargs):
    run.clirun(mcdef, **kwargs)


if __name__ == "__main__":
    clirun()
