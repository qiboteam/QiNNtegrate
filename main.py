"""
    Main script to launch the examples and benchmarks
"""

from argparse import ArgumentParser

from qinntegrate.target import valid_target
from qinntegrate.optimization import launch_optimization
from qinntegrate import quanting


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--xmin", help="Integration limit xi", nargs="+", type=float)
    parser.add_argument("--xmax", help="Integration limit xf", nargs="+", type=float)

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
    circ_parser.add_argument("--nqubits", help="Number of qubits for the VQE", default=3, type=int)
    circ_parser.add_argument("--layers", help="Number of layers for the VQE", default=2, type=int)

    opt_parser = parser.add_argument_group("Optimization definition")
    opt_parser.add_argument(
        "--maxiter", help="Maximum number of iterations", default=int(1e3), type=int
    )
    opt_parser.add_argument("--npoints", help="Training points", default=int(5e2), type=int)

    args = parser.parse_args()

    # Construct the target function
    target_fun = args.target(parameters=args.parameters, ndim=args.ndim)

    # Construct the observable to be trained
    observable = quanting.BaseVariationalObservable(nqubits=args.nqubits, nlayers=args.layers)

    # Prepare the integration limits
    xmin = args.xmin
    xmax = args.xmax
    if xmin is None:
        xmin = [0.0] * target_fun.ndim
    if xmax is None:
        xmax = [1.0] * target_fun.ndim

    # And... integrate!
    best_p = launch_optimization(observable, target_fun, max_iterations=args.maxiter)

    target_result, err = target_fun.integral(xmin, xmax)
    print(f"The target result for the integral of [{target_fun}] is {target_result:.4} +- {err:.4}")

    # Let's see the how this did...
    observable.set_parameters(best_p)
    primitive_minus = observable.execute(xmin)
    primitive_plus = observable.execute(xmax)
    res = primitive_plus - primitive_minus
    print(f"And our trained result is {res:.4}")
