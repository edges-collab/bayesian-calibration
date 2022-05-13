import click
import simulation_exploration_utils as utils
from run_alan_cal_mcmc import run_lk
import run_alan_cal_mcmc as nonsim
from typing import Any

main = click.Group()
from functools import partial

LABEL_FORMATS = (
    "c{cterms:d}_w{wterms:d}_cf{cterms_fit:d}_wf{wterms_fit:d}_smooth{smooth:d}_tns{tns_width:d}_var-{variance}",
    "c{cterms:d}_w{wterms:d}_cf{cterms_fit:d}_wf{wterms_fit:d}_smooth{smooth:d}_tns{tns_width:d}_var-{variance}_data-{as_data}",

)
FOLDER = "sim_cal_chains"

get_label = partial(nonsim.get_label, label_format=LABEL_FORMATS[-1])

def get_label(label_format=None, **kwargs):
    label_format = label_format or LABEL_FORMATS[-1]
    
    kwargs['as_data'] = [d[0] for d in kwargs['as_data']]

    return label_format.format(**kwargs)

    
def get_kwargs(label: str) -> dict[str, Any]:
    out = nonsim.get_kwargs(label, label_formats=LABEL_FORMATS)
    if out["variance"] != "data":
        out["variance"] = float(out["variance"])

    dct = {'o': 'open', 'h': 'hot_load', 'a': 'ambient', 's': 'short'}
    as_data = [dct[thing] for thing in out['as_data']]
    out['as_data'] = as_data

    return out


@main.command()
@click.option("-s", "--seed", default=1234)
@click.option("--smooth", default=32)
@click.option("--variance", default=None, type=float)
@click.option("-c", "--cterms", default=6)
@click.option("-w", "--wterms", default=5)
@click.option("--resume/--no-resume", default=False)
@click.option("-o/-O", "--optimize/--no-optimize", default=True)
@click.option("--clobber/--no-clobber", default=False)
@click.option("-C", "--fit-cterms", default=6)
@click.option("-W", "--fit-wterms", default=5)
@click.option("--prior-width", default=10.0)
@click.option("--tns-width", default=100)
@click.option("--as-data", multiple=True, type=click.Choice(['open', 'short', 'ambient', 'hot_load']))
def run(
    seed,
    smooth,
    variance,
    cterms,
    wterms,
    resume,
    optimize,
    clobber,
    fit_cterms,
    fit_wterms,
    prior_width,
    tns_width,
    as_data
):
    if variance is None:
        variance = "data"

    label = get_label(
        smooth=smooth,
        variance=variance,
        cterms=cterms,
        wterms=wterms,
        cterms_fit=fit_cterms,
        wterms_fit=fit_wterms,
        tns_width=tns_width,
        as_data=as_data
    )
    lk = utils.get_cal_likelihood(
        utils.get_calobs(cterms=cterms, wterms=wterms, smooth=smooth),
        seed=seed,
        variance=variance,
        cterms=fit_cterms,
        wterms=fit_wterms,
        tns_width=tns_width,
        as_data=as_data
    )

    run_lk(
        lk,
        label,
        resume,
        nlive_fac=100,
        optimize=optimize,
        clobber=clobber,
        prior_width=prior_width,
        folder=FOLDER
    )


if __name__ == "__main__":
    run()
