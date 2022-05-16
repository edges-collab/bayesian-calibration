import click
from alan_data_utils import get_likelihood
import simulation_exploration_utils as utils
from run_alan_cal_mcmc import run_lk
from typing import Any
import run_alan_precal_mcmc as precal

main = click.Group()
from functools import partial

LABEL_FORMATS = (
    "c{cterms:d}_w{wterms:d}_cf{cterms_fit:d}_wf{wterms_fit:d}_smooth{smooth:d}_tns{tns_width:d}_var-{variance}_s11{s11_sys}_nscale{nscale:02d}_ndelay{ndelay:02d}",
)
FOLDER = "sim_cal_chains"

get_label = precal.get_label
get_likelihood = partial(precal.get_likelihood, as_sim=('hot_load', 'ambient', 'short', 'open'))    

def get_kwargs(label: str) -> dict[str, Any]:
    out = precal.get_kwargs(label)
    if out["variance"] != "data":
        out["variance"] = float(out["variance"])
    return out


@main.command()
@click.option("--resume/--no-resume", default=False)
@click.option("-s", "--smooth", default=8)
@click.option("-p", "--tns-width", default=500)
@click.option("-n", "--nlive-fac", default=100)
@click.option("-o", "--optimize", type=click.Choice(['none', 'dual_annealing', 'basinhopping'], case_sensitive=False), default='basinhopping')
@click.option("--clobber/--no-clobber", default=False)
@click.option("--seed", default=1234)
@click.option("--tns-mean-zero/--est-tns", default=True)
@click.option('--ignore-sources', multiple=True, type=click.Choice(['short', 'open','hot_load', 'ambient']))
@click.option("--log-level", default='info', type=click.Choice(['info', 'debug', 'warn', 'error']))
@click.option("--s11-sys", multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient', 'rcv']))
@click.option("--run-mcmc/--no-mcmc", default=True)
@click.option("--opt-iter", default=10)
@click.option("--ndelay", default=1, type=int)
@click.option("--nscale", default=1, type=int)
def run(
    **kwargs
):
    precal.clirun(**kwargs)



if __name__ == "__main__":
    run()
