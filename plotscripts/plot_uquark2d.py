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

from quanting import qPDF_v2
from target import UquarkPDF2d


def get_primitive(obs, xmin, xmax, scaled_q):
    upper_limit = [xmax, scaled_q]
    lower_limit = [xmin, scaled_q]
    upper_val = obs.execute_with_x(upper_limit)
    lower_val = obs.execute_with_x(lower_limit)
    return upper_val - lower_val


def relative_error(x, y):
    return 100 * np.abs(x - y) / x


parser = ArgumentParser()
parser.add_argument("--output_folder", help="Output folder where to read the data from", type=Path)

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
    "--n_points", help="Number of q points for which we predict the integral", default=100, type=int
)

args = parser.parse_args()

json_file = args.output_folder / "args.json"
weight_file = args.output_folder / "best_p.npy"

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

print("Evaluating targets")
for qscaled in qscaled_points:
    res, error = updf.integral(xmin, xmax, qscaled, verbose=False, exact=True)
    target_vals.append(res)
    errors.append(error)

for exp in range(args.n_predictions):
    for i, qscaled in enumerate(qscaled_points):
        # only the first experiment
        circuit_vals[exp][i] = get_primitive(qpdf, xmin, xmax, qscaled)

target_vals = np.array(target_vals)
circuit_vals = np.array(circuit_vals)

# calculate matrix of relative errors
rr = np.zeros((args.n_predictions, args.n_points))

for exp in range(args.n_predictions):
    rr[exp] = relative_error(target_vals, circuit_vals[exp])

# mean and std of the predictions
mean_circuit_vals = np.mean(circuit_vals, axis=0)
std_circuit_vals = np.std(circuit_vals, axis=0)

# mean and std of the relative errors
mean_rr = np.mean(rr, axis=0)
std_rr = np.std(rr, axis=0)

plt.figure(figsize=(8, 5))
plt.grid(True)
plt.subplots(2, 1, sharex=True)

plt.subplot(2, 1, 1)
plt.plot(q2points, target_vals, color="blue", alpha=0.7, label="True result")
plt.plot(q2points, mean_circuit_vals, color="red", alpha=0.7, label="Approximation")
plt.fill_between(
    q2points,
    mean_circuit_vals - std_circuit_vals,
    mean_circuit_vals + std_circuit_vals,
    color="red",
    hatch="//",
    alpha=0.15,
)
plt.ylabel(r"$\int_0^{1} xu(x, q) dx$")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(q2points, mean_rr, color="black", alpha=0.8, label="Error")
plt.fill_between(
    q2points, mean_rr - std_rr, mean_rr + std_rr, color="black", hatch="//", alpha=0.15
)
plt.ylabel("% error")
plt.legend()
plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
plt.subplots_adjust(wspace=0, hspace=0)
plt.xlabel(r"$Q^2$ (GeV$^2$)")

plt.savefig("test.png")
