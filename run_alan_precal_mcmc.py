from __future__ import annotations
import click
import alan_data_utils as utils
from yabf.samplers.polychord import polychord
from edges_cal.modelling import LinLog
from pathlib import Path
from yabf import run_map
import sys
import parse
from typing import Dict
import numpy as np
from p_tqdm import p_map
from glob import glob
from getdist import loadMCSamples
from rich.logging import RichHandler
from yabf.core import mpi
import logging
from rich.console import Console
from rich.rule import Rule
from scipy import optimize as opt
from numdifftools import Hessian
from edges_cal import receiver_calibration_func as rcf
import attr
from typing import Any
import time
import yaml
from datetime import datetime
from scipy.stats import lognorm, norm
import pickle
from edges_cal.modelling import Polynomial, UnitTransform
import git
import edges_io
import edges_cal
import edges_analysis
import edges_estimate


cns = Console(width=200)
log = logging.getLogger(__name__)
log.addHandler(RichHandler(rich_tracebacks=True, console=cns))

main = click.Group()

LABEL_FORMATS = (
    "smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}",
    "smooth{smooth:02d}_tns{tns_width:04d}_ign[{ignore_sources}]_sim[{as_sim}]_s11{s11_sys}_nscale{nscale:02d}_ndelay{ndelay:02d}_unw-{unweighted}_cnf{cable_noise_factor:d}",
)

FOLDER = "alan_precal"
DEFAULT_KWARGS = {
    'tns_width': 500, 
    'ignore_sources': (), 
    's11_sys': (), 
    "nscale": 1, 
    "ndelay": 1, 
    'unweighted': False, 
    'cable_noise_factor': 1
}

def make_s11_sys(name, nscale, ndelay, scale_max, delay_max):
    out = dict(
        scale_model = Polynomial(n_terms=nscale, transform=UnitTransform(range=(50, 100))),
        delay_model = Polynomial(n_terms=ndelay, transform=UnitTransform(range=(50, 100)))
    )
    
    scale_params = {f"{name}_logscale_{i}": {'min': np.log10(1 - scale_max), 'max': np.log10(1 + scale_max), 'determines': f'logscale_{i}'} for i in range(nscale)}
    delay_params = {f"{name}_delay_{i}": {'min': -delay_max, 'max': delay_max, 'determines': f'delay_{i}'} for i in range(ndelay)}
    
    return {
        **out,
        **scale_params,
        **delay_params
    }
     

def define_s11_systematics(s11_sys: tuple[str], nscale: int, ndelay: int, scale_max: float=1e-2, delay_max: float=10):
    s11_systematic_params = {}
    for src in s11_sys:
        s11_systematic_params[src] = make_s11_sys(src, nscale=nscale, ndelay=ndelay, scale_max=scale_max, delay_max=delay_max)

    return s11_systematic_params

def get_likelihood(
    smooth, tns_width, est_tns=None, ignore_sources=(), as_sim=(), 
    s11_sys=(), nscale=1, ndelay=1, unweighted=False, cable_noise_factor=1,
):
    calobs = utils.get_calobs(cterms=6, wterms=5, smooth=smooth)
    s11_systematic_params = define_s11_systematics(s11_sys, nscale=nscale, ndelay=ndelay)
    return utils.get_cal_lk(
        calobs, tns_width=tns_width, est_tns=est_tns, 
        ignore_sources=ignore_sources, as_sim=as_sim, 
        s11_systematic_params=s11_systematic_params,
        sig_by_sigq=not unweighted, sig_by_tns=not unweighted,
        cable_noise_factor=cable_noise_factor,
    )


def get_label(label_format=None, **kwargs):
    s11_sys = kwargs.pop("s11_sys", ()) or ()

    label_format = label_format or LABEL_FORMATS[-1]
    return label_format.format(s11_sys=s11_sys, **kwargs)


def get_kwargs(label: str) -> dict[str, Any]:
    yaml_file = Path('outputs') / FOLDER / label / 'bayescal.lkargs.yaml'
    if yaml_file.exists():
        with open(FOLDER / label / 'bayescal.lkargs.yaml', 'r') as fl:
            kw = yaml.safe_load(fl)
    else:
        for fmt in LABEL_FORMATS[::-1]:
            kw = parse.parse(fmt, label)
            if kw:
                kw = kw.named
                break
            else:
                kw = {}

        for k, v in kw.items():
            # Convert booleans
            if v in ("True", "False"):
                kw[k] = (v == "True")

            # Convert tuples of strings.    
            if isinstance(v, str) and v[0] == '(' and v[-1] == ')':
                if len(v)>2:
                    kw[k] = tuple(v[1:-1].replace("'", "").replace(' ','').split(','))
                else:
                    kw[k] = ()

    return {**DEFAULT_KWARGS, **kw}


def get_samples_file(**kwargs):
    return f"outputs/{FOLDER}/{get_label(**kwargs)}/bayescal.txt"


def get_all_cal_curves(mcsamples, nthreads=1, force=False):
    folder = Path('outputs') / FOLDER
    kwargs = get_kwargs(Path(mcsamples.root).parent.name)
    if force:
        outfile = folder / (get_label(**kwargs) / "bayescal_blobs.npz")
        log.warning(f"Overwriting {outfile} since force=True")
    
    else:
        for fmt in LABEL_FORMATS:
            label = get_label(fmt, **kwargs)
            outfile = Path(folder) / f"{label}_blobs.npz"

            if outfile.exists():
                return dict(np.load(outfile))

        log.warning(f"{outfile} doesn't exist, so producing it.")
        
    freq = np.linspace(50, 100, 200)

    lk = get_likelihood(**kwargs)
    
    # Get the EQUAL WEIGHTS samples!
    samples = np.genfromtxt(mcsamples.root + "_equal_weights.txt")[
        :, 2 : (2 + len(lk.partial_linear_model.child_active_params))
    ]

    del lk

    nper_thread = len(samples) // nthreads
    last_n = nper_thread
    if len(samples) % nthreads:
        nper_thread += 1
        last_n = len(samples) - (nthreads - 1) * nper_thread

    def do_stuff(thread):
        cal_lk = get_likelihood(**kwargs)

        nn = last_n if thread == nthreads - 1 else nper_thread

        tcal = dict(
            tns=np.zeros((nn, len(freq))),
            tload=np.zeros((nn, len(freq))),
            tunc=np.zeros((nn, len(freq))),
            tcos=np.zeros((nn, len(freq))),
            tsin=np.zeros((nn, len(freq))),
            params=np.zeros((nn, cal_lk.partial_linear_model.linear_model.n_terms)),
        )
        start = thread * nper_thread

        for i, sample in enumerate(samples[start : start + nn]):
            out = cal_lk.get_cal_curves(params=sample, freq=freq, sample=True)
            for name, val in out.items():
                tcal[name][i] = val

        return tcal

    out = p_map(do_stuff, range(nthreads), num_cpus=nthreads)

    out_dict = {"samples": samples, "freq": freq}
    for i, name in enumerate(out[0]):
        out_dict[name] = np.concatenate([o[name] for o in out])

    np.savez(outfile, **out_dict)

    return out_dict


def get_completed_runs(read: bool = False):
    pth = Path('outputs') / FOLDER

    all_runs = [p for p in sorted(pth.glob('*'))]
    
    completed_runs = []
    for run in all_runs:
        if (run / 'bayescal.paramnames').exists():
            completed_runs.append(run)

    if read:
        return {fl.name: loadMCSamples(str(fl)) for fl in completed_runs}
    else:
        return completed_runs


def get_recalibrated_src_temps(blobs, root, calobs, nthreads=1, force=False):
    if Path(root+"_src_temps.npz").exists() and not force:
        return np.load(root+ "_src_temps.npz")

    n = len(blobs["samples"])

    nper_thread = n // nthreads
    last_n = nper_thread
    if n % nthreads:
        nper_thread += 1
        last_n = n - (nthreads - 1) * nper_thread

    freq = calobs.freq.freq
    tload = calobs.t_load
    lna_s11 = calobs.lna_s11
    tload_ns = calobs.t_load_ns
    loads = list(calobs._loads.keys())


    s11corr = calobs.s11_correction_models
    uncal_temps = {
        name: calobs.t_load_ns * load.spectrum.averaged_Q + calobs.t_load
        for name, load in calobs._loads.items()
    }

    for name in calobs.io.s11.simulators:
        load = calobs.new_load(name)

        s11corr[name] = load.s11_model(freq)
        uncal_temps[name] = calobs.t_load_ns * load.spectrum.averaged_Q + calobs.t_load

    def put(thread):
        kwargs = get_kwargs(Path(root).name)
        cal_lk = get_likelihood(**kwargs)

        nn = last_n if thread == nthreads - 1 else nper_thread
        start = thread * nper_thread
        model = cal_lk.nw_model.linear_model.model

        cal_temps = np.zeros((len(loads), nn, len(freq)))

        for i in range(nn):
            tnsp = blobs["samples"][start + i]
            pset = blobs["params"][start + i]

            tns = cal_lk.t_ns_model.model.model(x=freq, parameters=tnsp)
            off = tload - model.get_model("tload", x=freq, parameters=pset)
            t_unc = model.get_model("tunc", x=freq, parameters=pset)
            t_cos = model.get_model("tcos", x=freq, parameters=pset)
            t_sin = model.get_model("tsin", x=freq, parameters=pset)

            for j, (name, load_s11) in enumerate(s11corr.items()):
                a, b = rcf.get_linear_coefficients(
                    load_s11,
                    lna_s11,
                    sca=tns / tload_ns,
                    off=off,
                    t_unc=t_unc,
                    t_cos=t_cos,
                    t_sin=t_sin,
                    t_load=tload,
                )
                cal_temps[j, i] = a * uncal_temps[name] + b

        return cal_temps

    out = p_map(put, range(nthreads), num_cpus=nthreads)

    out = np.concatenate(out, axis=1)
    out = {name: out[i] for i, name in enumerate(calobs._loads)}

    np.savez(root + '_src_temps.npz', **out)
    return out

def get_recalibrated_src_temp_best(mcsamples, calobs, labcal):
    loads = {**calobs.loads}
    for name in calobs.metadata['io'].s11.simulators:
        loads[name] = calobs.new_load(name, io_obj = calobs.metadata['io'])

    uncal_temps = {
        name: calobs.t_load_ns * load.spectrum.averaged_Q + calobs.t_load
        for name, load in loads.items()
    }
    uncal_vars = {
        name: calobs.t_load_ns**2 * load.spectrum.variance_Q / load.spectrum.n_integrations
        for name, load in loads.items()
    }

    kwargs = get_kwargs(Path(mcsamples.root).name)
    cal_lk = get_likelihood(**kwargs)

    cal_temps = {}
    # NOTE: loglikes in mcsamples is actually -2*loglike :/
    pbest = mcsamples.samples[np.argmax(-mcsamples.loglikes)]
    n = len(cal_lk.partial_linear_model.child_active_params)

    for name, load in loads.items():
        cal_temps[name] = {}

        a, b = cal_lk.get_linear_coefficients(freq=calobs.freq.freq.to_value("MHz"), labcal=labcal, load=load, params=pbest[:n])
        cal_temps[name]['a'] = a
        cal_temps[name]['b'] = b
        cal_temps[name]['cal_temp'] = a*uncal_temps[name] + b
        cal_temps[name]['cal_var'] = a**2 * uncal_vars[name]
        cal_temps[name]['uncal_var'] = uncal_vars[name]
        
    return cal_temps


def run_lk(
    resume=False,
    nlive_fac=100,
    clobber=False,
    raise_on_prior=True,
    optimize=True,
    truth=None,
    prior_width=10,
    set_widths: bool=False,
    run_mcmc: bool=True,
    opt_iter: int = 10,
    **lk_kwargs
):
    label = get_label(**lk_kwargs)
    lk = get_likelihood(**lk_kwargs)

    repo = git.Repo(str(Path(__file__).parent.absolute()))

    root = 'bayescal'
    folder = Path('outputs') / FOLDER / label
    out_txt = folder / (root + '.txt')
    out_yaml = folder / (root + '.meta.yml')

    if mpi.am_single_or_primary_process:

        if not folder.exists():
            folder.mkdir(parents=True)

        cns.print(f"[bold]Running [blue]{label}")
        cns.print(f"Output will go to '{folder}'")

        cns.print(
            f"[bold]Fiducial Parameters[/]: {[p.fiducial for p in lk.partial_linear_model.child_active_params]}"
        )
        t = time.time()
        lnl, derived = lk.partial_linear_model()
        t1 = time.time()
        cns.print(f"[bold]Fiducial likelihood[/]: {lnl}")
        for nm, d in zip(lk.partial_linear_model.child_derived, derived):
            cns.print(f"\t{nm if isinstance(nm, str) else nm.__name__}: {d}")

        cns.print(f"Took {t1 - t:1.2e} seconds to evaluate likelihood.")

        if not resume and out_txt.exists():
            if not clobber:
                sys.exit(
                    f"Run with label '{label}' already exists. Use --resume to resume it, or delete it."
                )
            else:
                all_files = folder.glob(f"{root}*")
                flstring = "\n\t".join(str(fl.name) for fl in all_files)
                log.warning(f"Removing following files:\n{flstring}")
                for fl in all_files:
                    fl.unlink()

        # Write out the likelihood args
        with open(folder / (root + '.lkargs.yaml'), 'w') as fl:
            yaml.dump(lk_kwargs, fl)

    if optimize and (not resume or not out_txt.exists() or not run_mcmc):
        # Only run the optimizatino if we're not just resuming an MCMC
        lk = optimize_lk(lk, truth, prior_width,  folder, root, dual_annealing = optimize == 'dual_annealing', niter=opt_iter, set_widths=set_widths)
        

    if mpi.am_single_or_primary_process:
        if resume and out_txt.exists() and out_yaml.exists():
            with open(out_yaml, 'r') as fl:
                yaml_args = yaml.safe_load(fl)
        else:
            yaml_args = {
                'start_time': time.time(), 
                'start_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'optimize': optimize,
                'prior_width': prior_width,
                'set_widths': set_widths,
                'githash': repo.head.ref.commit.hexsha + ('.dirty' if repo.is_dirty() else ''),
                'edges-io': edges_io.__version__,
                'edges-cal': edges_cal.__version__,
                'edges-analysis': edges_analysis.__version__,
                'edges-estimate': edges_estimate.__version__,
            }

            with open(out_yaml, 'w') as fl:
                yaml.dump(yaml_args, fl)

    if run_mcmc:
        poly = polychord(
            save_full_config=False,
            likelihood=lk.partial_linear_model,
            output_dir=folder,
            output_prefix=root,
            sampler_kwargs=dict(
                nlive=nlive_fac * len(lk.partial_linear_model.child_active_params),
                read_resume=resume,
                feedback=2,
            ),
        )

        def time_dumper(live, dead, logweights, logZ, logZerr):
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            this_time = time.time()
            cns.print(f"{now}: {(this_time - time_dumper.last_call) / 60} min since last dump.")
            time_dumper.last_call = this_time

        t = time.time()
        time_dumper.last_call = t
        samples = poly.sample(dumper=time_dumper)
        cns.print(f"Sampling took {(time.time() - t)/60/60:1.3} hours.")
        samples.saveAsText(f"{folder}/{label}")

        if mpi.am_single_or_primary_process:
            yaml_args['end_time'] = t
            yaml_args['end_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(out_yaml, 'w') as fl:
                yaml.dump(yaml_args, fl)

        # Do some basic diagnostics
        means = samples.getMeans()
        stds = np.sqrt(samples.getVars())
        err = False
        for i, p in enumerate(lk.partial_linear_model.child_active_params):
            if means[i] - 5 * stds[i] < p.min or means[i] + 5 * stds[i] > p.max:
                err = True
                log.error(
                    f"Parameter '{p.name}' has posterior out of bounds. Posterior: {means[i]} += {stds[i]}, bounds = {p.min} - {p.max}."
                )

        if err and raise_on_prior:
            raise MCMCBoundsError("Parameter posteriors are out of prior bounds!")

        return samples, err

def optimize_lk(lk, truth, prior_width, folder, label, dual_annealing: bool=False, niter: int=10, set_widths=False):
    if mpi.am_single_or_primary_process:
        cns.print(f"Optimizing with {niter} global iterations using {'dual_annealing' if dual_annealing else 'basinhopping'}...", end='')
        t = time.time()

        minima = []

        def callback(x, f, accept):
            minima.append((x, f))

        if dual_annealing:
            opt_res = run_map(
                lk.partial_linear_model,
                dual_annealing_kw={"maxiter": niter, "callback": callback},
            )
        else:
            opt_res = run_map(
                lk.partial_linear_model,
                basinhopping_kw={"niter": niter, "callback": callback},
            )

            if opt_res.minimization_failures > 0:
                log.warning(
                        f"There were {opt_res.minimization_failures} minimization failures!"
                    )

            if not opt_res.lowest_optimization_result.success:
                log.warning(
                        f"The lowest optimization was not successful! Message: {opt_res.lowest_optimization_result.message}"
                    )

        def obj(x):
            out = -lk.partial_linear_model.logp(params=x)
            # cns.print(x, out)
            if np.isnan(out) or np.isinf(out):
                log.warning(f"Hessian got NaN/inf value for parameters: {x}, {out}")
            return out
        
        cns.print("Computing Hessian...")

        hess = Hessian(obj, base_step=0.1, step_ratio=3, num_steps=30)(opt_res.x)
        cns.print("Hessian: ", hess)
        cov = np.linalg.inv(hess)
        cns.print("Covariance: ", cov)
        std = np.sqrt(np.diag(cov))
        widths = prior_width * std
        resx = opt_res.x

        cns.print("[bold]Estimated Parameters: [/]")
        for p, r, s in zip(lk.partial_linear_model.child_active_params, resx, std):
            cns.print(f"\t{p.name:>14}: {r:1.3e} +- {s:1.3e}")

        f = [f for _, f in minima]
        minf = min(f)
        _minima = [m for m in minima if m[1] < (minf + 100)]

        if len(_minima) == 1:
            log.warning(f"Only one minima was any good, got: {f}")
        else:
            for i, param in enumerate(lk.partial_linear_model.child_active_params):
                rxx = resx[i]
                ww = widths[i]

                xx  = [x[i] for x, _ in _minima]
                
                if any((np.abs(xxx - rxx) > ww and (ff < minf + 100)) for xxx, ff in zip(xx, f)):
                    log.error(
                        f"For '{param.name}', got minima at {xx} "
                        f"when it should have been between {rxx-ww} and {rxx+ww}."
                        f"Corresponding -lks: {[f for _, f in _minima]} (high likelihoods omitted)."
                    )
                    
        if truth is not None and np.any(np.abs(truth - resx) > 3 * std):
            raise RuntimeError(
                    "At least one of the estimated parameters was off the truth by > 3σ"
                )
        cns.print(f" done in {time.time() - t:.2f} seconds.")

        # Write out the results in a pickle.
        with open(folder / (label + '.map'), 'wb') as fl:
            pickle.dump(
                {
                    'optres': opt_res,
                    'cov': cov,
                    'minima': minima,
                },
                fl
            )

    else:
        resx = None
        widths = None

    widths = mpi.mpi_comm.bcast(widths, root=0)
    resx = mpi.mpi_comm.bcast(resx, root=0)

    if set_widths:
        # NOTE: If this is true, then the final evidence one calculates should be modified
        #       like so: log(Z) ~ log(Z_polychord) - n*log(f),
        #       where n is the number of dimensions that have been compressed by some
        #       width and f is the factor by which each dimension is compressed.
        #       More generally, it's \Sum(log(f_i)) where the sum goes over each dimension.
        #       HOWEVER, this only works if the posterior is essentially zero outside
        #       the actual prior range chosen for polychord. 
        #       For a prior_width of 10, this is a very good approximation for dimensions
        #       well beyond 30. 
        new_tns_params = attr.evolve(
                lk.t_ns_params, fiducial=resx, min=resx - widths, max=resx + widths
            )

        lk = attr.evolve(lk, t_ns_params=new_tns_params)

    else:
        for i, p in enumerate(lk.t_ns_params.get_params()):
            if resx[i] - widths[i] < p.min or resx[i] + widths[i] > p.max:
                raise ValueError(f"You need to set Tns[{i}] to have greater width. At least {resx[i] + widths[i]}")

    return lk


class MCMCBoundsError(ValueError):
    pass


@main.command()
@click.option("--resume/--no-resume", default=False)
@click.option("-s", "--smooth", default=8)
@click.option("-p", "--tns-width", default=500)
@click.option("-n", "--nlive-fac", default=100)
@click.option("-o", "--optimize", type=click.Choice(['none', 'dual_annealing', 'basinhopping'], case_sensitive=False), default='basinhopping')
@click.option("--clobber/--no-clobber", default=False)
@click.option("--set-widths/--no-set-widths", default=False)
@click.option("--tns-mean-zero/--est-tns", default=True)
@click.option('--ignore-sources', multiple=True, type=click.Choice(['short', 'open','hot_load', 'ambient']))
@click.option('--as-sim', multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient']))
@click.option("--log-level", default='info', type=click.Choice(['info', 'debug', 'warn', 'error']))
@click.option("--s11-sys", multiple=True, type=click.Choice(['short', 'open', 'hot_load', 'ambient', 'rcv']))
@click.option("--run-mcmc/--no-mcmc", default=True)
@click.option("--opt-iter", default=10)
@click.option("--unweighted/--weighted", default=False)
@click.option("--cable-noise-factor", default=1, type=int)
@click.option("--ndelay", default=1, type=int)
@click.option("--nscale", default=1, type=int)
def run(
    **kwargs
):
    clirun(**kwargs)


def clirun(**kwargs):
    log_level = kwargs.pop("log_level")
    root_logger = logging.getLogger('yabf')
    root_logger.setLevel(log_level.upper())
    root_logger.addHandler(RichHandler(rich_tracebacks=True, console=cns))
    
    optimize = kwargs.pop('optimize').lower()

    if optimize == 'none':
        optimize = None

    tns_mean_zero = kwargs.pop('tns_mean_zero')
    
    for k, v in kwargs.items():
        if isinstance(v, list):
            kwargs[k] = tuple(v)

    run_lk(
        optimize=optimize,
        est_tns=np.zeros(6) if tns_mean_zero else None,
        **kwargs
    )



if __name__ == "__main__":
    run()
