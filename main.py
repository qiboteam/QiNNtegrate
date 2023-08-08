"""
    Main script to launch the examples and benchmarks
"""
from argparse import ArgumentParser, ArgumentTypeError
import copy
import json
from pathlib import Path
import tempfile

from matplotlib import pyplot as plt
import numpy as np

from qinntegrate.optimization import available_optimizers, launch_optimization
from qinntegrate.quanting import available_ansatz, generate_ansatz_pool
from qinntegrate.target import available_targets

TARGETS = list(available_targets.keys())
ANSATZS = list(available_ansatz.keys())
OPTIMIZ = list(available_optimizers.keys())
nicered = "#ff6150"


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


def ratio(pred, target, eps=1e-3):
    tts = np.where(np.abs(target) > eps, target, np.sign(target) * eps)
    ccs = np.where(np.abs(pred) > eps, pred, np.sign(pred) * eps)
    return np.abs(tts / ccs)


def plot_uquark(predictor, target, xmin, xmax, output_folder, npoints=50):
    """Plot botht he predictor and the target"""
    xmin = np.array(xmin)
    xmax = np.array(xmax)

    for d in range(target.ndim):
        xaxis_scale = "log"  # target.dimension_scale(d)
        # Create a linear space in the dimension we are plotting
        xlin = np.linspace(xmin[d], xmax[d], npoints)

        if xaxis_scale == "log":
            # change to log
            xlin = np.logspace(np.log10(xmin[d]), np.log10(xmax[d]), npoints)

        for i in range(target.ndim):
            # For every extra dimension do an extra plot so that we have more random points
            # in the other dimensions

            # Select a random point in the other dimensions
            xran_origin = np.random.rand(target.ndim) * (xmax - xmin) + xmin

            ytrue = []
            all_xs = []

            for xx in xlin:
                xran = copy.deepcopy(xran_origin)
                xran[d] = xx
                ytrue.append(target(xran))
                all_xs.append(xran)

            ypred = predictor.vectorized_forward_pass(all_xs)

            if target.ndim == 2:
                # when there is only 2 dimensions only one variable is fixed
                # and so we can actually write the numerical value
                other_d = (d + 1) % 2
                fixed_name = target.dimension_name(other_d)
                tag = f"{fixed_name}={xran[other_d]:.2}"
            else:
                tag = f"n{i}"


            rr = ratio(ypred, ytrue)

            # plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.5, 4.5*6/8), gridspec_kw={"height_ratios": [5, 2]})

            ax1.plot(
                xlin,
                ypred,
                label=f"Approximation {tag}",
                linewidth=2.5,
                alpha=0.9,
                ls="-",
                color="#ff6150",
            )
            ax1.plot(
                xlin,
                np.stack(ytrue),
                label=f"Target $u$-quark {tag}",
                linewidth=1.5,
                alpha=0.7,
                ls="--",
                color="black",
            )
            ax1.grid(False)
            ax1.set_xscale(xaxis_scale)
            ax1.set_ylabel(r"$u\,f(x)$")
            ax1.set_title(rf"$u$-quark PDF fit", fontsize=12)
            fig.legend(bbox_to_anchor=(0.55, 0.58), framealpha=1)

            ax2.plot(xlin, rr, color=nicered, lw=2.5, alpha=0.9)
            ax2.hlines(1, 1e-4, 1, color="black", ls="--", lw=1.5, alpha=0.7)
            ax2.grid(False)
            ax2.set_xscale(xaxis_scale)
            ax2.set_xlabel(r"x")
            ax2.set_ylabel("Ratio")
            ax2.set_ylim(0.97, 1.03)

            plt.rcParams['xtick.bottom'] = True
            plt.rcParams['ytick.left'] = True
            
            fig.subplots_adjust(wspace=0, hspace=0)

        
        plt.savefig(output_folder / f"uquark1d.pdf", bbox_inches="tight")
        plt.close()


def plot_integrand(predictor, target, xmin, xmax, output_folder, npoints=50):
    if "quark" in str(target):
        return plot_uquark(predictor, target, xmin, xmax, output_folder, npoints=50)

    xmin = np.array(xmin)
    xmax = np.array(xmax)

    for d in range(target.ndim):
        xaxis_name = target.dimension_name(d)
        xaxis_scale = target.dimension_scale(d)
        # Create a linear space in the dimension we are plotting
        xlin = np.linspace(xmin[d], xmax[d], npoints)

        if xaxis_scale == "log":
            # change to log
            xlin = np.logspace(np.log10(xmin[d]), np.log10(xmax[d]), npoints)

        for i in range(target.ndim):
            # For every extra dimension do an extra plot so that we have more random points
            # in the other dimensions

            # Select a random point in the other dimensions
            xran_origin = np.random.rand(target.ndim) * (xmax - xmin) + xmin

            ytrue = []
            all_xs = []

            for xx in xlin:
                xran = copy.deepcopy(xran_origin)
                xran[d] = xx
                ytrue.append(target(xran))
                all_xs.append(xran)

            ypred = predictor.vectorized_forward_pass(all_xs)

            if target.ndim == 2:
                # when there is only 2 dimensions only one variable is fixed
                # and so we can actually write the numerical value
                other_d = (d + 1) % 2
                fixed_name = target.dimension_name(other_d)
                tag = f"{fixed_name}={xran[other_d]:.2}"
            else:
                tag = f"n={i}"

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]

            plt.plot(
                xlin,
                ypred,
                label=f"Approximation {tag}",
                linewidth=2.5,
                alpha=0.6,
                ls="-",
                color=color,
            )
            plt.plot(
                xlin,
                np.stack(ytrue),
                label=f"Target {tag}",
                linewidth=1.5,
                alpha=0.8,
                ls="--",
                color=color,
            )

        plt.grid(True)
        plt.xscale(xaxis_scale)
        # plt.title(f"Integrand fit, dependence on {xaxis_name}")
        plt.xlabel(rf"${xaxis_name}$")
        plt.xlabel(r"x")
        plt.ylabel(r"$u\,f(x)$")
        plt.savefig(output_folder / f"output_plot_d{d+1}.pdf")
        plt.close()


def _generate_limits(xmin, xmax, dimensions=1):
    """Generate the lists of limits to evaluate the primitive at

    For the dimensions that are not integrated the upper limits
    will be used as value with which the circuit will be called

    Parameters
    ---------
        xmin: list of inferior limits (one per dimension)
        xmax: list of superior limits
        dimensions: int
            dimensions over which the integral is taken

    Returns
    -------
        limits: list of 1d arrays with one value per dimension
        signs: list of same size with + or -
    """
    limits = [np.array([])]
    signs = [1.0]
    for i, (xm, xp) in enumerate(zip(xmin, xmax)):
        next_l = []
        next_s = []
        for curr_l, curr_s in zip(limits, signs):
            if i < dimensions:
                next_l.append(np.concatenate([curr_l, [xm]]))
                next_s.append(-curr_s)

            next_l.append(np.concatenate([curr_l, [xp]]))
            next_s.append(curr_s)
        limits = next_l
        signs = next_s

    return limits, signs


def _generate_integration_x(xmin, xmax, padding=False, npoints=int(5e2)):
    """Generate a set of random poitns within the integration limits"""
    xmin = np.array(xmin)
    xmax = np.array(xmax)
    if padding:
        xdelta = xmax - xmin
        xmin -= 0.1 * xdelta
        xmax += 0.1 * xdelta
    return np.random.rand(npoints, len(xmin)) * (xmax - xmin) + xmin


def error_over_runs(results, errors):  # not being used at the moment
    """Calculate error of the measurements provided with their errors"""
    N = len(results)
    var = np.var(results)
    return var**2 / N + (1 / N**2) * np.sum(errors**2)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--xmin", help="Integration limit xi", nargs="+", type=float, default=[0.0])
    parser.add_argument("--xmax", help="Integration limit xf", nargs="+", type=float, default=[1.0])
    parser.add_argument("-o", "--output", help="Output folder", type=Path, default=None)
    parser.add_argument("-l", "--load", help="Load initial parameters from", type=Path)
    parser.add_argument(
        "-j", "--jobs", help="Number of processes to utilize (default 4)", type=int, default=4
    )

    target_parser = parser.add_argument_group("Target function")
    target_parser.add_argument(
        "--target", help=f"Select target function, available: {TARGETS}", type=str, default="sin1d"
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
        default="reuploading",
        type=str,
    )
    # Circuit features
    circ_parser.add_argument(
        "--nqubits", help="Number of qubits for the VQE", default=1, type=check_qbits
    )
    circ_parser.add_argument("--layers", help="Number of layers for the VQE", default=2, type=int)
    circ_parser.add_argument("--nshots", help="Number of shots for each < Z > evaluation", type=int)

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
        type=str,
        default="CMA",
    )
    opt_parser.add_argument(
        "--nruns", help="Number of times the optimization is repeated", default=1, type=int
    )

    args = parser.parse_args()

    # Update args
    a_target = valid_target(args.target)
    a_ansatz = valid_ansatz(args.ansatz)
    a_optimizer = valid_optimizer(args.optimizer)

    # doesn't make sense to perform more than one run if simulation is exact
    if args.nshots is None and args.nruns != 1:
        raise ValueError(
            "It is useless to set nruns > 1 if exact simulation is performed. Please set a number of shots or set nruns to be equal to 1."
        )

    if args.output is None:
        # Generate a temporary output folder
        output_folder = Path(tempfile.mkdtemp())
    else:
        output_folder = args.output
    output_folder.mkdir(exist_ok=True)
    print(output_folder)

    # Construct the target function
    target_fun = a_target(parameters=args.parameters, ndim=args.ndim)
    observable = generate_ansatz_pool(
        a_ansatz,
        nqubits=args.nqubits,
        nlayers=args.layers,
        ndim=args.ndim,
        nshots=args.nshots,
        nprocesses=args.jobs,
        nderivatives=target_fun.nderivatives,
    )

    xmin = args.xmin
    xmax = args.xmax

    # Check whether what the user gave makes sense and otherwise crash
    if len(xmin) != target_fun.ndim:
        if len(xmin) != 1:
            raise ValueError(
                "Please give either as many `xmin` lower limits as dimensions or just one (which will be used for all dimensions"
            )
        xmin = xmin * target_fun.ndim

    if len(xmax) != target_fun.ndim:
        if len(xmax) != 1:
            raise ValueError(
                "Please give either as many `xmax` upper limits as dimensions or just one (which will be used for all dimensions"
            )
        xmax = xmax * target_fun.ndim

    xarr = _generate_integration_x(xmin, xmax, padding=args.padding)

    if target_fun.override:
        xmin = target_fun.xmin
        xmax = target_fun.xmax
        if target_fun.xgrid is None:
            xarr = _generate_integration_x(xmin, xmax, padding=args.padding)
        else:
            xarr = np.array(target_fun.xgrid).reshape(-1, target_fun.ndim)

    print(f" > Using {xmin} as lower limit of the integral")
    print(f" > Using {xmax} as upper limit of the integral")

    if args.load:
        initial_p = np.load(args.load)

    # to collect all simulation results
    simulation_results = np.zeros(args.nruns, dtype=np.float64)

    # execute the experiment nruns times
    for i in range(args.nruns):
        # initialize the problem to initial_p
        if args.load:
            observable.set_parameters(initial_p)

        # optimize
        best_p = launch_optimization(
            xarr,
            observable,
            target_fun,
            a_optimizer,
            max_iterations=args.maxiter,
            normalize=not args.absolute,
        )

        # Let's see how this integral did
        observable.set_parameters(best_p)

        # Prepare all combinations of limits
        limits, signs = _generate_limits(xmin, xmax, dimensions=observable.nderivatives)

        # And... integrate!
        res = 0.0
        print(limits)
        for int_limit, sign in zip(limits, signs):
            res += sign * observable.execute_with_x(int_limit)

        simulation_results[i] = res
        print(f"Result for exp {i+1}/{args.nruns}: {res:.4}", end="\r")

    # mean and std over the simulation results
    results_mean = np.mean(np.asarray(simulation_results))
    results_std = np.std(np.asarray(simulation_results))
    print(f"Average trained result is {results_mean:.4} +- {results_std:.4}")

    # print theoretical results
    target_result, err = target_fun.integral(xmin, xmax)
    print(f"The target result for the integral of [{target_fun}] is {target_result:.4} +- {err:.4}")

    plot_uquark(observable, target_fun, xmin, xmax, output_folder)

    print(f"Saving results to {output_folder}\n")

    best_p_path = output_folder / "best_p.npy"
    np.save(best_p_path, best_p)
    # And save also the parameters that we've used!
    arg_path = output_folder / "args.json"
    opts = copy.copy(args.__dict__)
    # Drop what we don't need
    opts.pop("output")
    opts.pop("load")
    # And change some
    opts["xmin"] = xmin
    opts["xmax"] = xmax
    opts["FinalResult"] = res
    opts["TargetResult"] = target_result
    json.dump(opts, arg_path.open("w", encoding="utf-8"), indent=True)

    # Force close the pool in case (tensorflow?) is leaving it open
    del observable
