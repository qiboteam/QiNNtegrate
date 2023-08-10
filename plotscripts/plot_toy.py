"""Script to create a plot of the integral of the toy model for the paper for a given value of alpha
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np

# put this in
sys.path.append("../qinntegrate")
sys.path.append("../")

from quanting import GoodScaling

from main import _generate_limits, nicered


def get_primitive_target(parameters, xmin, xmax, xx, differential_in_d=1, alpha=0.0):
    """Return the exact value of the integral of cos(parameters•x + alpha)
    The integral is performed in as many dimension as thge length of xmin/xmax
    """
    intd = len(xmin)

    limits, signs = _generate_limits(xmin, xmax, dimensions=len(xmin))

    parf = np.prod(parameters) / parameters[differential_in_d]
    m = (-1) ** (intd // 2) * parf
    if intd % 2 == 0:
        fun = lambda x: m * np.cos(x)
    else:
        fun = lambda x: m * np.sin(x)

    res = 0.0
    for int_limit, sign in zip(limits, signs):
        xarr = int_limit.tolist()
        xarr.insert(differential_in_d - 1, xx)
        arg = np.sum(np.array(parameters[1:]) * np.array(xarr)) + alpha
        res += sign * fun(arg)
    return res


def get_primitive_circuit(obs, xmin, xmax, xx, differential_in_d=1, alpha=0.0):
    """Return the circuit (obs) evaluated at the given values of xmin/xmax/xx/alpha
    Note that the circuit will only be derivated on d
    """
    limits, signs = _generate_limits(xmin, xmax, dimensions=len(xmin))

    res = 0.0
    for int_limit, sign in zip(limits, signs):
        xarr = int_limit.tolist()
        xarr.insert(differential_in_d - 1, xx)
        xarr.append(alpha)
        res += sign * obs.forward_pass(xarr, derivate_ds=[differential_in_d])
    return res


def ratio(pred, target, eps=1e-3):
    tts = np.where(np.abs(target) > eps, target, np.sign(target) * eps)
    ccs = np.where(np.abs(pred) > eps, pred, np.sign(pred) * eps)
    return np.abs(tts / ccs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_folder", help="Output folder where to read the data from", type=Path
    )
    parser.add_argument("--xmin", help="Lower integration limit", type=float, default=0)
    parser.add_argument("--xmax", help="Upper integration limit", type=float, default=np.pi / 2)
    parser.add_argument("--npoints", help="How many points to plot", type=int, default=15)
    parser.add_argument("--differential", help="Dimension for the diff dist.", default=1, type=int)
    parser.add_argument("--alpha", help="Fixed value fo alpha", default=0.0, type=float)
    parser.add_argument(
        "--error",
        help="Compute error bands using all weights found in the folder",
        action="store_true",
    )

    args = parser.parse_args()

    npoints = args.npoints

    json_file = args.data_folder / "args.json"

    # Get all possible weights in this output folder
    weights = [np.load(args.data_folder / "best_p.npy")]
    files = []
    if args.error:
        for weight_file in args.data_folder.glob("best_p_*.npy"):
            files.append(weight_file)
            weights.append(np.load(weight_file))

    json_info = json.loads(json_file.read_text(encoding="utf-8"))
    ndim = json_info["ndim"]

    dim_diff = args.differential
    if dim_diff > (ndim - 1):
        raise ValueError("Please select a dimension that was integrated over")

    ansatz = GoodScaling(json_info["nqubits"], json_info["layers"], ndim=ndim)
    params = json_info["parameters"]

    xmin = args.xmin
    xmax = args.xmax

    xmin_l = [xmin] * (ndim - 2)
    xmax_l = [xmax] * (ndim - 2)

    xlin = np.linspace(xmin, xmax, npoints)
    a = args.alpha

    target_vals = np.array(
        [get_primitive_target(params, xmin_l, xmax_l, xx, dim_diff, alpha=a) for xx in xlin]
    )

    all_circuit_vals = []
    for replica_weights in weights:
        ansatz.set_parameters(replica_weights)
        circuit_vals = [
            get_primitive_circuit(ansatz, xmin_l, xmax_l, xx, dim_diff, alpha=a) for xx in xlin
        ]
        all_circuit_vals.append(circuit_vals)
    all_circuit_vals = np.stack(all_circuit_vals)

    circuit_vals = np.mean(all_circuit_vals, axis=0)
    circuit_errs = np.std(all_circuit_vals, axis=0)
    c_p = circuit_vals + circuit_errs
    c_m = circuit_vals - circuit_errs

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(4.5, 4.5 * 6 / 8), gridspec_kw={"height_ratios": [5, 2]}
    )

    ax1.set_title(rf"$\alpha = {a}$", fontsize=12)
    ax1.plot(
        xlin, circuit_vals, alpha=0.9, label="Approximation", color="#ff6150", linewidth=2.5, ls="-"
    )
    ax1.fill_between(xlin, c_m, c_p, color=nicered, hatch="//", alpha=0.25)
    ax1.plot(
        xlin, target_vals, alpha=0.7, label="Target result", color="black", linewidth=1.5, ls="-."
    )

    ax1.set_ylabel(rf"$I(\alpha={a}, x_{dim_diff})$")
    ax1.grid(False)

    rr = ratio(circuit_vals, target_vals)
    r1 = ratio(c_p, target_vals)
    r2 = ratio(c_m, target_vals)

    rp = np.maximum(r1, r2)
    rm = np.minimum(r1, r2)
    fig.legend(bbox_to_anchor=(0.85, 0.88), framealpha=1)

    ax2.plot(xlin, rr, color=nicered, alpha=0.9, lw=2.5)
    ax2.fill_between(xlin, rm, rp, color=nicered, hatch="//", alpha=0.35)
    ax2.hlines(1, 0, 3, color="black", alpha=0.7, ls="-.", lw=1.5)
    ax2.set_ylim(0.93, 1.07)
    ax2.set_ylabel("Ratio")
    ax2.grid(False)
    ax2.set_xlabel(rf"$x_{dim_diff}$")

    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    fig.subplots_adjust(wspace=0, hspace=0)

    output_file = f"cos_on_x{dim_diff}_alpha{a}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    print(f"Plot saved to {output_file}, average ratio: {np.mean(rr)}")
