"""Script to create a plot of
    Int[ x*u(x, q) ] vs q
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from matplotlib import pyplot as plt
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
parser.add_argument("output_folder", help="Output folder where to read the data from", type=Path)

args = parser.parse_args()

json_file = args.output_folder / "args.json"
weight_file = args.output_folder / "best_p.npy"

json_info = json.loads(json_file.read_text(encoding="utf-8"))

qpdf = qPDF_v2(1, json_info["layers"], ndim=2)
qpdf.set_parameters(np.load(weight_file))
updf = UquarkPDF2d(ndim=2)

xmin = 1e-3
xmax = 0.70
qscaled_points = np.linspace(0, 1, 40)
q2points = (qscaled_points * (updf._max_q - updf._min_q) + updf._min_q) ** 2

target_vals = []
circuit_vals = []
errors = []

for qscaled in qscaled_points:
    res, error = updf.integral(xmin, xmax, qscaled, verbose=False, exact=True)
    target_vals.append(res)
    errors.append(error)
    circuit_vals.append(get_primitive(qpdf, xmin, xmax, qscaled))

target_vals = np.array(target_vals)
circuit_vals = np.array(circuit_vals)
plt.figure(figsize=(8, 5))
plt.title("Marginalization of the PDF integral")
plt.grid(True)
plt.subplots(2, 1, sharex=True)

plt.subplot(2, 1, 1)
plt.plot(q2points, target_vals, color="blue", alpha=0.7, label="True result")
plt.plot(q2points, circuit_vals, color="red", alpha=0.7, label="Approximation")
plt.ylabel(r"$\int_0^{1} xu(x, q) dx$")
plt.legend()

plt.subplot(2, 1, 2)
rr = relative_error(target_vals, circuit_vals)
plt.plot(q2points, rr, color="black", alpha=0.5, label="Error")
plt.ylabel("% error")
plt.legend()

plt.xlabel(r"$Q^2$ (GeV$^2$)")

plt.savefig("test.png")
