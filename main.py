"""
    Main script to launch the examples and benchmarks
"""

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from qinntegrate.target import valid_target
from qinntegrate.optimization import launch_optimization
from qinntegrate import quanting


def check_qbits(var):
    nqbit = int(var)
    if nqbit < 2:
        raise ArgumentTypeError(
            "At least 2 qbits are needed in order for entanglement to be active"
        )
    return nqbit


def plot_integrand(predictor, target, xmin, xmax, npoints=int(1e3)):
    """Plot botht he predictor and the target"""
    xlin = np.linspace(xmin, xmax, npoints)

    ytrue = []
    ypred = []
    for xx in xlin:
        ytrue.append(target(xx))
        ypred.append(predictor.forward_pass(xx))

    plt.plot(xlin[:, 0], ytrue, label="Target function")
    plt.plot(xlin[:, 0], ypred, label="Simulation")
    plt.legend()
    plt.savefig("output.pdf")


def _generate_limits(xmin, xmax):
    """Generate the lists of limits to evaluate the primitive at
    Parameters
    ---------
        xmin: list of inferior limits (one per dimension)
        xmax: list of superior limits

    Returns
    -------
        limits: list of 1d arrays with one value per dimension
        signs: list of same size with + or -
    """
    limits = [np.array([])]
    signs = [1.0]
    for xm, xp in zip(xmin, xmax):
        next_l = []
        next_s = []
        for curr_l, curr_s in zip(limits, signs):
            next_l.append(np.concatenate([curr_l, [xm]]))
            next_s.append(curr_s)

            next_l.append(np.concatenate([curr_l, [xp]]))
            next_s.append(-curr_s)
        limits = next_l
        signs = next_s

    return limits, signs


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--xmin", help="Integration limit xi", nargs="+", type=float)
    parser.add_argument("--xmax", help="Integration limit xf", nargs="+", type=float)
    parser.add_argument("-o", "--output", help="Output folder", type=Path, default=Path("output/"))
    parser.add_argument("-l", "--load", help="Load initial parameters from", type=Path)

    target_parser = parser.add_argument_group("Target function")
    target_parser.add_argument(
        "--target", help="Select target function", type=valid_target, default=valid_target("sin1d")
    )
    target_parser.add_argument(
        "--parameters",
        help="List of parameters for the target functions",
        nargs="+",
        type=float,
        default=(),
    )
    target_parser.add_argument("--ndim", help="Number of dimensions", type=int, default=1)

    # Circuit parameters
    circ_parser = parser.add_argument_group("Circuit definition")
    circ_parser.add_argument(
        "--nqubits", help="Number of qubits for the VQE", default=3, type=check_qbits
    )
    circ_parser.add_argument("--layers", help="Number of layers for the VQE", default=2, type=int)

    opt_parser = parser.add_argument_group("Optimization definition")
    opt_parser.add_argument(
        "--maxiter", help="Maximum number of iterations", default=int(1e3), type=int
    )
    opt_parser.add_argument("--npoints", help="Training points", default=int(5e2), type=int)
    opt_parser.add_argument(
        "--padding", help="Train the function beyond the integration limits", action="store_true"
    )

    args = parser.parse_args()

    # Construct the target function
    target_fun = args.target(parameters=args.parameters, ndim=args.ndim)

    # Construct the observable to be trained
    observable = quanting.BaseVariationalObservable(
        nqubits=args.nqubits, nlayers=args.layers, ndim=args.ndim
    )

    # Prepare the integration limits
    xmin = args.xmin
    xmax = args.xmax
    if xmin is None:
        xmin = [0.0] * target_fun.ndim
    if xmax is None:
        xmax = [1.0] * target_fun.ndim

    # And... integrate!
    if args.load:
        initial_p = np.load(args.load)
        observable.set_parameters(initial_p)

    best_p = launch_optimization(
        observable, target_fun, max_iterations=args.maxiter, padding=args.padding
    )

    target_result, err = target_fun.integral(xmin, xmax)
    print(f"The target result for the integral of [{target_fun}] is {target_result:.4} +- {err:.4}")

    # Let's see how this integral did
    observable.set_parameters(best_p)

    # Prepare all combinations of limits
    limits, signs = _generate_limits(xmin, xmax)

    res = 0.0
    for int_limit, sign in zip(limits, signs):
        res += sign * observable.execute_with_x(int_limit)

    print(f"And our trained result is {res:.4}")

    if args.ndim == 1:
        plot_integrand(observable, target_fun, xmin, xmax)

    if args.output is not None:
        args.output.mkdir(exist_ok=True)
        np.save(args.output / "best_p.npy", best_p)
