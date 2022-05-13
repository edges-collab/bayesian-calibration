import click
import simulation_exploration_utils as utils
from yabf.samplers.polychord import polychord

main = click.Group()

@main.command()
@click.option('-s', '--seed', default=1234)
def run(seed):
    lk = utils.get_likelihood(utils.fid_labcal, seed=seed)

    poly = polychord(
        save_full_config=False,
        likelihood=lk.partial_linear_model,
        output_dir='polychord_chains',
        sampler_kwargs=dict(
            nlive=4096,
            read_resume=True,
            feedback = 2
        )
    )

    samples = poly.sample()
    print("Sampled")
    samples.saveAsText("polychord_chains/")

if __name__ == '__main__':
    run()