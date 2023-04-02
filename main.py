"""
    Main script to launch the examples and benchmarks
"""
import json
import copy
import tempfile
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from qinntegrate.target import available_targets
from qinntegrate.optimization import launch_optimization, available_optimizers
from qinntegrate.quanting import available_ansatz

TARGETS = list(available_targets.keys())
ANSATZS = list(available_ansatz.keys())
OPTIMIZ = list(available_optimizers.keys())


def check_qbits(var):
    nqbit = int(var)
    if nqbit < 1:
        raise ArgumentTypeError("Number of qubits must be positive")
    return nqbit


def valid_this(val_raw, options, name=""):
    """Ensures that the option val_raw is included in the available options"""
    # Make sure that everything is lowercase
    val = val_raw.lower()
    options = {i.lower(): k for i, k in options.items()}
    if val not in options:
        ava = list(options.keys())
        raise ArgumentTypeError(f"{name} {val_raw} not allowed, allowed options are {ava}")
    return options[val]


def valid_target(val_raw):
    return valid_this(val_raw, available_targets, "Target")


def valid_ansatz(val_raw):
    """Ensures that the selected ansatz exists
    Note that this does not check whether the number of dimensions/qbits/etc is acceptable
    acyclic graphs are beyond of the scope of this project...
    """
    return valid_this(val_raw, available_ansatz, "Ansatz")


def valid_optimizer(val_raw):
    return valid_this(val_raw, available_optimizers, "Optimizer")


def plot_integrand(predictor, target, xmin, xmax, output_folder, npoints=int(1e3)):
    """Plot botht he predictor and the target"""
    xlin = np.linspace(xmin, xmax, npoints)

    ytrue = []
    ypred = []
    for xx in xlin:
        ytrue.append(target(xx))
        ypred.append(predictor.forward_pass(xx))

    plt.plot(xlin[:, 0], ytrue, label="Target function", alpha=0.7)
    plt.plot(xlin[:, 0], ypred, label="Simulation", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.savefig(output_folder / "output_plot.pdf")


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
            next_s.append(-curr_s)

            next_l.append(np.concatenate([curr_l, [xp]]))
            next_s.append(curr_s)
        limits = next_l
        signs = next_s

    return limits, signs


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--xmin", help="Integration limit xi", nargs="+", type=float)
    parser.add_argument("--xmax", help="Integration limit xf", nargs="+", type=float)
    parser.add_argument("-o", "--output", help="Output folder", type=Path, default=None)
    parser.add_argument("-l", "--load", help="Load initial parameters from", type=Path)

    target_parser = parser.add_argument_group("Target function")
    target_parser.add_argument(
        "--target",
        help=f"Select target function, available: {TARGETS}",
        type=valid_target,
        default=valid_target("sin1d"),
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
    # Circuit ansatz
    circ_parser.add_argument(
        "--ansatz",
        help=f"Circuit ansatz, please choose one among {ANSATZS}",
        default=valid_ansatz("base"),
        type=valid_ansatz,
    )
    # Circuit features
    circ_parser.add_argument(
        "--nqubits", help="Number of qubits for the VQE", default=3, type=check_qbits
    )
    circ_parser.add_argument("--layers", help="Number of layers for the VQE", default=2, type=int)

    opt_parser = parser.add_argument_group("Optimization definition")
    opt_parser.add_argument(
        "--maxiter",
        help="Maximum number of iterations (default 1000)",
        default=int(1e3),
        type=int,
    )
    opt_parser.add_argument(
        "--npoints", help="Training points (default 500)", default=int(5e2), type=int
    )
    opt_parser.add_argument(
        "--padding",
        help="Train the function beyond the integration limits",
        action="store_true",
    )
    opt_parser.add_argument(
        "--absolute",
        help="Don't normalize MSE by the size of the integrand",
        action="store_true",
    )
    opt_parser.add_argument(
        "--optimizer",
        help=f"Optimizers, available options: {OPTIMIZ}",
        type=valid_optimizer,
        default=valid_optimizer("CMA"),
    )
    opt_parser.add_argument(
        "--exact",
        help="When active, the derivative is not taken and thus the training is C -> target",
        action="store_true",
    )
    opt_parser.add_argument("--tol_error", help="Tolerance on the error function (for the optimizer that accept it)", default=1e-5, type=float)

    args = parser.parse_args()

    if args.output is None:
        # Generate a temporary output folder
        output_folder = Path(tempfile.mkdtemp())
    else:
        output_folder = args.output
    output_folder.mkdir(exist_ok=True)

    # Construct the target function
    target_fun = args.target(parameters=args.parameters, ndim=args.ndim)

    observable = args.ansatz(nqubits=args.nqubits, nlayers=args.layers, ndim=args.ndim, derivative = not args.exact)

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
        observable,
        target_fun,
        args.optimizer,
        max_iterations=args.maxiter,
        padding=args.padding,
        normalize=not args.absolute,
        tol_error=args.tol_error
    )

    if args.exact:
        res = target_result = "Derivative not taken"
    else:
        target_result, err = target_fun.integral(xmin, xmax)
        print(
            f"The target result for the integral of [{target_fun}] is {target_result:.4} +- {err:.4}"
        )

        # Let's see how this integral did
        observable.set_parameters(best_p)

        # Prepare all combinations of limits
        limits, signs = _generate_limits(xmin, xmax)

        res = 0.0
        for int_limit, sign in zip(limits, signs):
            res += sign * observable.execute_with_x(int_limit)

        print(f"And our trained result is {res:.4}")

    if args.ndim == 1:
        plot_integrand(observable, target_fun, xmin, xmax, output_folder)

    print(f"Saving results to {output_folder}")

    best_p_path = output_folder / "best_p.npy"
    np.save(best_p_path, best_p)
    # And save also the parameters that we've used!
    arg_path = output_folder / "args.json"
    opts = copy.copy(args.__dict__)
    # Drop what we don't need
    opts.pop("output")
    opts.pop("load")
    # And change some
    opts["target"] = str(target_fun)
    opts["xmin"] = xmin
    opts["xmax"] = xmax
    opts["FinalResult"] = res
    opts["TargetResult"] = target_result
    opts["ansatz"] = str(observable)
    opts["optimizer"] = str(args.optimizer)
    json.dump(opts, arg_path.open("w", encoding="utf-8"), indent=True)
