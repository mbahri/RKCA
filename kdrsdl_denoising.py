import click
import numpy as np

from models import RkcaOrderTwoAdmm

import experiments


@click.command()
@click.option("--exp-name", type=click.Choice(["yale", "facade"]), required=True)
@click.option(
    "--noise-level", type=click.FLOAT, default=0.1, required=False, show_default=True
)
@click.option("--r", type=click.INT, default=None, required=False, show_default=True)
@click.option(
    "--lamb", type=click.FLOAT, default=None, required=False, show_default=True
)
@click.option(
    "--tol", type=click.FLOAT, default=1e-7, required=False, show_default=True
)
@click.option(
    "--maxiter", type=click.INT, default=150, required=False, show_default=True
)
@click.option("--rho", type=click.FLOAT, default=1.2, required=False, show_default=True)
@click.option(
    "--alpha-a", type=click.FLOAT, default=1, required=False, show_default=True
)
@click.option(
    "--alpha-b", type=click.FLOAT, default=1, required=False, show_default=True
)
@click.option(
    "--alpha", type=click.FLOAT, default=1e-1, required=False, show_default=True
)
@click.option("--mu", type=click.FLOAT, default=None, required=False, show_default=True)
@click.option(
    "--mu-bar", type=click.FLOAT, default=np.inf, required=False, show_default=True
)
@click.option("--convergence-core-R", is_flag=True)
def main(
    exp_name,
    noise_level,
    r,
    lamb,
    tol,
    maxiter,
    rho,
    alpha_a,
    alpha_b,
    alpha,
    mu,
    mu_bar,
    convergence_core_r,
):
    if exp_name == "yale":
        print(f"Loading data with noise level {noise_level}")
        GT, X = experiments.yale_sp(noise_level)
        pre_processing = experiments.pre_process_no_op
        post_processing = experiments.post_process_no_op

        # Index of the slice to visualize
        i = 1
        visualizer = lambda GT, X, L, E: experiments.visualize_slices(
            GT[i], X[i], L[i], E[i]
        )

    elif exp_name == "facade":
        print(f"Loading data with noise level {noise_level}")
        GT, X = experiments.facade_sp(noise_level)
        # RKCA expect the first dimension to be the slice index
        pre_processing = experiments.pre_process_color
        post_processing = experiments.post_process_color

        visualizer = lambda GT, X, L, E: experiments.visualize_slices(GT, X, L, E)

    # -------------------------------------------------------------

    RKCA = RkcaOrderTwoAdmm(
        X,
        r=r,
        lambda_=lamb,
        tol=tol,
        maxiter=maxiter,
        rho=rho,
        alpha_a=alpha_a,
        alpha_b=alpha_b,
        alpha=alpha,
        mu=mu,
        mu_bar=mu_bar,
        convergence_core_R=convergence_core_r,
        post_processing=post_processing,
        pre_processing=pre_processing,
    )
    converged, iters, errors = RKCA.fit()

    experiments.plot_convergence(errors)

    print("Getting reconstructions from sparse codes.")
    X, L, E = RKCA.get_reconst()

    visualizer(GT, X, L, E)
    print("Done.")


if __name__ == "__main__":
    main()
