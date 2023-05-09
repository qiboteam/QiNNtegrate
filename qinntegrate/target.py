r"""
    Library of target fucntions to test and benchmark the algoritm

    All target functions in this library are initialized with an array of parameters 
    of arbitrary length defining the function and expose the methods:
    
    __call__(self, [x1, x2, ..., xn]) -> f(\vec{x})

    integral(self, [xmin1, xmin2, ..., xminn], [xmax1, xmax2, ..., xmaxn]) -> \Int_{\vec{xmin}}^{\vec{xmax}} dxf(x)
"""
from abc import abstractmethod
from pathlib import Path

import numpy as np
from scipy.integrate import nquad
from scipy.interpolate import interp1d

try:
    from pdfflow import mkPDF
except ModuleNotFoundError:

    def mkPDF(*args, **kwargs):
        raise ModuleNotFoundError("Please install `pdfflow`, `pip install pdfflow`")


class TargetFunction:
    """
    Base class protocol for all target functions
    This class contains a number of parameterS:
        max_par: maximum number of parameters allowed by the target function
        max_ndim: maximum number of dimensions allowed by the target function
        override: whether the target function should override the xgrid used for training
    """

    max_par = 1e10
    max_ndim = 20
    override = False

    def __init__(self, parameters=(), ndim=1):
        self._parameters = np.array(parameters)
        self.ndim = ndim
        if len(parameters) > self.max_par:
            raise ValueError(f"This target function accepts a maximum of {self.max_par} parameters")
        if ndim > self.max_ndim:
            raise ValueError(
                f"This target function accepts a maximum of {self.max_ndim} to integrate"
            )
        print(f"Preparing {self} with d={self.ndim}")

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
        return self.__class__.__name__

    # When override == True, the target function should also implement the following properties
    @property
    def xmin(self):
        return [0.0]

    @property
    def xmax(self):
        return [1.0]

    @property
    def xgrid(self):
        pass


class Sin1d(TargetFunction):
    """1 dimensional sin function:
    y = sin(a1*x + a2)
    """

    max_par = 3
    max_ndim = 1

    def build(self):
        self._a0 = self._parameters[0] if len(self._parameters) > 0 else 1.0
        self._a1 = self._parameters[1] if len(self._parameters) > 1 else 1.0
        self._a2 = self._parameters[2] if len(self._parameters) > 2 else 0.0

    def __call__(self, xarr):
        x = xarr[0]
        return np.sin(self._a1 * x + self._a2) * self._a0

    def __repr__(self):
        return "1d sin"


class Cosnd(TargetFunction):
    """Cosine of polynomial in x
    cos(a1*x1 + a2*x2 + ... + an+1)
    """

    def build(self):
        # Use the parameters in self._parameters for the first a_n
        # and the rest fill it with ones
        missing_par = (self.ndim + 2) - len(self._parameters)
        if missing_par > 0:
            fill_one = np.ones(missing_par)
            fill_one[-1] = 0
            self._parameters = np.concatenate([self._parameters, fill_one])

    def __call__(self, xarr):
        arg = np.sum(np.array(xarr) * self._parameters[1:-1]) + self._parameters[-1]
        return np.cos(arg) * self._parameters[0]

    def __repr__(self):
        return f"cos{self.ndim}d"


class Sind(Cosnd):
    def __call__(self, xarr):
        arg = np.sum(np.array(xarr) * self._parameters[:-1]) + self._parameters[-1]
        return np.sin(arg)

    def __repr__(self):
        return f"sin{self.ndim}d"


class LepageTest(TargetFunction):
    """Function used in Lepage's Vegas paper https://inspirehep.net/files/6f1e72c3ed9265314819355759f96e54"""

    max_par = 1

    def build(self):
        a = 0.1
        if self._parameters:
            a = self._parameters[0]

        self._a2 = np.power(a, 2)
        self._pref = np.power(1.0 / a / np.sqrt(np.pi), self.ndim)

    def __call__(self, xarr):
        res = 0
        for x in xarr:
            res += np.power(x - 0.5, 2) / self._a2
        return self._pref * np.exp(-res)

    def __repr__(self):
        return f"LepageTest {self.ndim}D"


class UquarkPDF(TargetFunction):
    """1d version of the uquark PDF
    Doesn't need to have a PDF interpolation library installed as
    the resutls for NNPDF4.0 are saved as an npz file"""

    max_par = 0
    max_ndim = 1
    override = True

    def build(self):
        npz_data = np.load(Path(__file__).parent / "uquark.npz")
        self._xgrid = npz_data.get("x")
        self._uvals = npz_data.get("y")
        self._eps = np.min(self._xgrid)
        self._uquark = interp1d(self._xgrid, self._uvals)

    def __call__(self, xarr):
        return self._uquark(xarr).squeeze()

    def __repr__(self):
        return f"xu(x)"

    @property
    def xmin(self):
        return [self._eps]

    @property
    def xmax(self):
        return [0.7]
        return [np.max(self._xgrid)]

    @property
    def xgrid(self):
        im = np.where(self._xgrid > self.xmin)[0][0]
        ip = np.where(self._xgrid < self.xmax)[0][-1]
        return self._xgrid[im:ip]


class UquarkPDF2d(TargetFunction):
    """Implementation of the u-quark PDF from NNPDF4.0
    It requires two dimensions (x,) and (Q,) and pdfflow needs to be installed
    """

    max_par = 0
    max_ndim = 2
    override = True

    _min_x = 1e-4
    _min_q = 6.65**2
    _max_x = 0.7
    _max_q = 16.65**2

    def build(self):
        if self.ndim < 2:
            raise ValueError("This target, uquark2d, needs 2 dimensions: x,q")

        nx = 70
        nq = 20
        x = np.concatenate(
            [
                np.logspace(np.log10(self._min_x), -1, nx // 2),
                np.linspace(0.1, self._max_x, nx // 2),
            ]
        )
        q = np.linspace(self._min_q, self._max_q, nq)

        xx, qq = np.meshgrid(x, q)
        self._xgrid = np.column_stack([xx.ravel(), qq.ravel()])

        self._pdf = mkPDF("nnpdf40/0", dirname=Path(__file__).parent)

    def __call__(self, xarr):
        x = xarr[0]
        q = xarr[1]
        return self._pdf.py_xfxQ2(2, [x], [q]).numpy()

    def __repr__(self):
        return f"xu(x)"

    def integral(self, xmin, xmax):
        npoints = 1000
        xgrid = np.linspace(xmin[0], xmax[0], npoints)
        q2 = xmin[1]
        print(f"Computing the integral for {q2=}")
        q2grid = [q2] * npoints
        vals = self._pdf.py_xfxQ2(2, xgrid, q2grid)
        xdelta = xmax[0] - xmin[0]
        return (np.average(vals) * xdelta, 0.0)

    @property
    def xmin(self):
        return [self._min_x, self._min_q]

    @property
    def xmax(self):
        return [self._max_x, self._max_q]

    @property
    def xgrid(self):
        return self._xgrid


available_targets = {
    "sin1d": Sin1d,
    "cosnd": Cosnd,
    "sind": Sind,
    "lepage": LepageTest,
    "uquark": UquarkPDF,
    "uquark2d": UquarkPDF2d,
}
