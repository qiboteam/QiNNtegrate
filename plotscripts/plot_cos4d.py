"""Script to create a plot of the integral of Cos4d marginalized over any of the 4 variables
    Int[  ] vs q
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

from main import _generate_limits


def _get_primitive(xmin, xmax, xx, derivate_n=1, parameters=None, obs=None):
    """
    Returns the integral of cosnd(parameters*x)*a0 for either the circuit surrogate
    or the actual primitive.
    """
    if (parameters is not None) and (obs is not None):
        raise ValueError("Not both a circuit and parameters can be given at the same time")

    limits, signs = _generate_limits(xmin, xmax, dimensions=len(xmin))
    res = 0.0
    for int_limit, sign in zip(limits, signs):
        xarr = int_limit.tolist()
        xarr.insert(derivate_n - 1, xx)
        if parameters is not None:
            arg = np.dot(xarr, parameters[1:])
            parf = np.prod(parameters) / parameters[derivate_n]
            res += sign * (-1) * np.sin(arg) / parf
        if obs is not None:
            res += sign * obs.forward_pass(xarr, derivate_ds=[derivate_n])

    return res


def get_primitive_circuit(obs, xmin, xmax, xx, derivate_n=1):
    return _get_primitive(xmin, xmax, xx, derivate_n=derivate_n, obs=obs)


def get_primitive_target(parameters, xmin, xmax, xx, derivate_n=1):
    return _get_primitive(xmin, xmax, xx, derivate_n=derivate_n, parameters=parameters)


def relative_error(x, y):
    return 100 * np.abs(x - y) / np.abs(x)


parser = ArgumentParser()
parser.add_argument("--output_folder", help="Output folder where to read the data from", type=Path)

parser.add_argument("--xmin", help="Lower integration limit", type=float, default=0)
parser.add_argument("--xmax", help="Upper integration limit", type=float, default=np.pi / 2)
parser.add_argument("--npoints", help="How many points to plot", type=int, default=15)
parser.add_argument(
    "--differential", help="Which dimension do you want to have the differential distribution for", default=1, type=int
)

args = parser.parse_args()
npoints = args.npoints

json_file = args.output_folder / "args.json"
weight_file = args.output_folder / "best_p.npy"

json_info = json.loads(json_file.read_text(encoding="utf-8"))
ndim = json_info["ndim"]
dim_diff = args.differential

if dim_diff > ndim:
    raise ValueError("Choose a smaller number")

ansatz = GoodScaling(json_info["nqubits"], json_info["layers"], ndim=ndim)
ansatz.set_parameters(np.load(weight_file))

parameters = json_info["parameters"]

target_vals = []
circuit_vals = []

xmin = args.xmin
xmax = args.xmax

xmin_limits = [xmin] * (ndim - 1)
xmax_limits = [xmax] * (ndim - 1)

xlin = np.linspace(xmin, xmax, npoints)

for xx in xlin:
    target_vals.append(get_primitive_target(parameters, xmin_limits, xmax_limits, xx, dim_diff))
    circuit_vals.append(get_primitive_circuit(ansatz, xmin_limits, xmax_limits, xx, dim_diff))

target_vals = np.array(target_vals)
circuit_vals = np.array(circuit_vals)
plt.figure(figsize=(8, 5))
plt.title("Marginalization of the PDF integral")
# plt.grid(True)
plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [5, 2]})

plt.subplot(2, 1, 1)
plt.plot(xlin, circuit_vals, alpha=0.6, label="Approximation", color="red", linewidth=2.5, ls="-")
plt.plot(xlin, target_vals, alpha=0.8, label="Target result", color="black", linewidth=1.5, ls="-.")
plt.ylabel(rf"$G(x_{dim_diff})$")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
rr = relative_error(target_vals, circuit_vals)
plt.plot(xlin, rr, color="blue", alpha=0.7, lw=2.5, label="Error")
plt.ylabel("% error")
# plt.legend()
plt.grid(True)
plt.subplots_adjust(wspace=0, hspace=0)
plt.xlabel(rf"$x_{dim_diff}$")

output_file = f"cosmarg_on_x{args.marginalize}.pdf"
plt.savefig(output_file)
print(f"Plot saved to {output_file}, average error: {np.mean(rr)}")
