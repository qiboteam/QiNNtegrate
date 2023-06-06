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
xmax = 0.7
qscaled_points = np.linspace(0, 1, 40)
q2points = (qscaled_points * (updf._max_q - updf._min_q) + updf._min_q) ** 2

target_vals = []
circuit_vals = []

for qscaled in qscaled_points:
    res, error = updf.integral(xmin, xmax, qscaled, verbose=False)
    target_vals.append(res)
    circuit_vals.append(get_primitive(qpdf, xmin, xmax, qscaled))


plt.grid(True)
plt.plot(q2points, target_vals, label="True result")
plt.plot(q2points, circuit_vals, label="Approximation")
plt.xlabel(r"$Q^2$ (GeV$^2$)")
plt.ylabel(r"$\int_0^{1} xu(x, q)$")
plt.legend()

plt.savefig("test.png")
