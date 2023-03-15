r"""
    Library of target fucntions to test and benchmark the algoritm

    All target functions in this library are initialized with an array of parameters 
    of arbitrary length defining the function and expose the methods:
    
    __call__(self, [x1, x2, ..., xn]) -> f(\vec{x})

    integral(self, [xmin1, xmin2, ..., xminn], [xmax1, xmax2, ..., xmaxn]) -> \Int_{\vec{xmin}}^{\vec{xmax}} dxf(x)
"""
from abc import abstractmethod
from argparse import ArgumentTypeError
import numpy as np
from scipy.integrate import nquad


def valid_target(val_raw):
    """Ensures that the selected function exists"""
    available_targets = {"sin1d": Sin1d, "cos1d": Cos1d}
    val = val_raw.lower()
    if val not in available_targets:
        ava = list(available_targets.keys())
        raise ArgumentTypeError(f"Target {val_raw} not allowed, allowed targets are {ava}")

    return available_targets[val]


class TargetFunction:
    max_par = 1e10
    max_ndim = 20

    def __init__(self, parameters=(), ndim=1):
        self._parameters = parameters
        self.ndim = ndim
        if len(parameters) > self.max_par:
            raise ValueError(f"This target function accepts a maximum of {self.max_par} parameters")
        if ndim > self.ndim:
            raise ValueError(
                f"This target function accepts a maximum of {self.max_ndim} to inetegate"
            )

        self.build()

    def build(self):
        pass

    @abstractmethod
    def __call__(self, xarr):
        pass

    def integral(self, xmin, xmax):
        """Integrate the target function using nquad"""
        fun = lambda *x: self(x)
        ranges = list(zip(xmin, xmax))
        return nquad(fun, ranges)

    def __repr__(self):
        return "Target Function"


class Sin1d(TargetFunction):
    """1 dimensional sin function:
    y = sin(a1*x + a2)
    """

    max_par = 2
    max_ndim = 1

    def build(self):
        self._a1 = self._parameters[0] if len(self._parameters) > 0 else 1.0
        self._a2 = self._parameters[1] if len(self._parameters) > 1 else 0.0

    def __call__(self, xarr):
        x = xarr[0]
        return np.sin(self._a1 * x + self._a2)

    def __repr__(self):
        return "1d sin"


class Cos1d(Sin1d):
    def __call__(self, xarr):
        x = xarr[0]
        return np.cos(self._a1 * x + self._a2)
