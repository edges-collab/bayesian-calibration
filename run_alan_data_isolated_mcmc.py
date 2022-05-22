import click
import alan_data_utils as utils
import alan_data as adata
from edges_cal.modelling import LinLog
from pathlib import Path
import numpy as np
from p_tqdm import p_map
import run_alan_precal_mcmc as precal
import run_alan_cal_mcmc as cal

main = click.Group()


precal.LABEL_FORMATS = (
    "c{cterms:02d}_w{wterms:02d}_smooth{smooth:02d}_fg{nterms_fg}_taufx{fix_tau}",
)

precal.FOLDER = "alan_field_isolated"
precal.DEFAULT_KWARGS={
    'cterms': 6,
    'wterms': 5, 
    'smooth': 32,
    'fg': 5,
    'fix_tau': False
}

def get_likelihood(
    nterms_fg, fix_tau, cterms, wterms, smooth,
):
    calobs = utils.get_calobs(cterms=cterms, wterms=wterms, smooth=smooth)
    labcal = utils.get_labcal(calobs)

    return utils.get_isolated_likelihood(
        labcal, 
        calobs,
        fsky=adata.sky_data['freq'],
        fg=LinLog(n_terms=nterms_fg), 
        eor=utils.make_absorption(adata.sky_data['freq'], fix=('tau',) if fix_tau else ()),
    )

precal.get_likelihood = get_likelihood

@main.command()
@cal.cterms
@cal.wterms
@precal.resume
@precal.smooth
@precal.nlive_fac
@precal.optimize
@precal.clobber
@precal.log_level
@precal.run_mcmc
@precal.opt_iter
@click.option('--nterms-fg', default=5)
@click.option('--fix-tau/--no-fix-tau', default=True)
def run(**kwargs):
    precal.clirun(**kwargs)

if __name__ == '__main__':
    run()