"""Script to create a plot of
    Int[ x*u(x, q) ] vs q
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

# put this in
sys.path.append("../qinntegrate")
sys.path.append("../")

from quanting import qPDF_v2
from target import UquarkPDF2d

from main import nicered


def get_primitive(obs, xmin, xmax, scaled_q):
    upper_limit = [xmax, scaled_q]
    lower_limit = [xmin, scaled_q]
    upper_val = obs.execute_with_x(upper_limit)
    lower_val = obs.execute_with_x(lower_limit)
    return upper_val - lower_val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_folder", help="Output folder where to read the data from", type=Path
    )

    # number of shots for each prediction
    parser.add_argument(
        "--nshots_predictions",
        help=f"Number of shots for evaluating predictions",
        default=1000,
        type=int,
    )
    # number of predictions for each q value
    parser.add_argument(
        "--n_predictions", help="Number of times we predict the integral", default=100, type=int
    )
    # number of q points
    parser.add_argument(
        "--n_points",
        help="Number of q points for which we predict the integral",
        default=100,
        type=int,
    )

    args = parser.parse_args()

    json_file = args.data_folder / "args.json"
    weight_file = args.data_folder / "best_p.npy"

    json_info = json.loads(json_file.read_text(encoding="utf-8"))

    qpdf = qPDF_v2(1, json_info["layers"], ndim=2, nshots=args.nshots_predictions)
    qpdf.set_parameters(np.load(weight_file))
    updf = UquarkPDF2d(ndim=2)

    xmin = 1e-3
    xmax = 0.70
    qscaled_points = np.linspace(0, 1, args.n_points)
    q2points = (qscaled_points * (updf._max_q - updf._min_q) + updf._min_q) ** 2

    target_vals = []
    # {experiments : raws, q_values : columns}
    circuit_vals = np.zeros((args.n_predictions, args.n_points))
    errors = []

    #     print("Evaluating targets")
    #     for qscaled, q2 in zip(qscaled_points, q2points):
    #         print(f"Integrating the real PDF for q={np.sqrt(q2)}")
    #         res, error = updf.integral(xmin, xmax, qscaled, verbose=False, exact=True)
    #         target_vals.append(res)
    #         errors.append(error)

    #     np.save(file=args.data_folder / "target_labels", arr=target_vals)
    #     np.save(file=args.data_folder / "target_errors", arr=errors)

    target_vals = np.load(args.data_folder / "target_labels.npy")
    target_errors = np.load(args.data_folder / "target_errors.npy")

    print("Calculating predictions using qinntegrate procedure.")
    print(f"The final prediction is the mean over {args.n_predictions}.")
    for exp in range(args.n_predictions):
        if (ie := exp + 1) % 10 == 0:
            print(f"Running experiment {ie:2d}/{args.n_predictions}", end="\r")
        for i, qscaled in enumerate(qscaled_points):
            # only the first experiment
            circuit_vals[exp][i] = get_primitive(qpdf, xmin, xmax, qscaled)

    target_vals = np.array(target_vals)
    circuit_vals = np.array(circuit_vals)

    # calculate matrix of relative errors
    rr = np.zeros((args.n_predictions, args.n_points))

    for exp in range(args.n_predictions):
        rr[exp] = np.abs(target_vals / circuit_vals[exp])

    # mean and std of the predictions
    mean_circuit_vals = np.mean(circuit_vals, axis=0)
    std_circuit_vals = np.std(circuit_vals, axis=0)

    # mean and std of the relative errors
    mean_rr = np.mean(rr, axis=0)
    std_rr = np.std(rr, axis=0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(4.5, 4.5 * 6 / 8), sharex=True, gridspec_kw={"height_ratios": [5, 3]}
    )
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True

    ax1.plot(q2points, mean_circuit_vals, color="#ff6150", alpha=0.9, lw=2.5, label="Approximation")
    ax1.fill_between(
        q2points,
        mean_circuit_vals - std_circuit_vals,
        mean_circuit_vals + std_circuit_vals,
        color="#ff6150",
        hatch="//",
        alpha=0.35,
    )
    ax1.plot(
        q2points, target_vals, color="black", alpha=0.7, lw=1.5, ls="--", label="Target result"
    )
    ax1.set_ylabel(r"$I_u(Q^2)$")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax1.set_title(r"Estimates of $I_u(Q^2)$")
    fig.legend(bbox_to_anchor=(0.9, 0.88), framealpha=1)

    ax2.plot(q2points, mean_rr, color=nicered, lw=2.5, alpha=0.9)
    ax2.fill_between(
        q2points, mean_rr - std_rr, mean_rr + std_rr, color=nicered, hatch="//", alpha=0.35
    )
    ax2.hlines(1, np.min(q2points), np.max(q2points), lw=1.5, color="black", ls="--", alpha=0.7)
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel(r"$Q^2$ (GeV$^2$)")
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig("uquark2d.pdf", bbox_inches="tight")
    print("Saved file to uquark2d.pdf")
