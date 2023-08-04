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
        raise ModuleNotFoundError("Please install `pdfflow`: `pip install pdfflow`")


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
                f"This target function accepts a maximum of {self.max_ndim} dimensions to integrate"
            )
        print(f"Preparing {self} with d={self.ndim}")

        self.build()

    def build(self):
        pass

    @abstractmethod
    def __call__(self, xarr):
        pass

    def integral(self, xmin, xmax, marginalized_vars=None):
        """Integrate the target function using nquad
        If there are marginalized variables, they can be given as list to marginalized_vars
        """
        if marginalized_vars is None:
            fun = lambda *x: self(x)
        else:
            fun = lambda *x: self(np.concatenate([marginalized_vars, x]))
        ranges = list(zip(xmin, xmax))
        return nquad(fun, ranges)

    def __repr__(self):
        return self.__class__.__name__

    def dimension_name(self, d):
        """Returns the name of dimensions d"""
        return f"x{d+1}"

    def dimension_scale(self, d):
        """Returns the scale of the dimension for plots (eg. log or linear)"""
        return "linear"

    # When override == True, the target function should also implement the following properties
    @property
    def xmin(self):
        return [0.0]

    @property
    def xmax(self):
        return [1.0]

    @property
    def xgrid(self):
        return None

    @property
    def nderivatives(self):
        return self.ndim


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


class CosndAlpha(TargetFunction):
    """Similar to Cosnd but with only two parameters available a1 and a2 which cannot be given
        cos(a2*x1 + x2 + ... + xn + a1)
    the code is prepared so that both a1 and a2 are parameters to be trained upon

    The minimum number of dimensions is 2 (a1 and x1)
    if 4 or more dimensions are selected then a2 will also be considered. No more parameters are considered.

    The training ranges are fixed to be between 0 and 3.5
    alpha instead is trainged between 0 and 10
    """

    max_par = 0
    override = True

    def build(self):
        if self.ndim < 2:
            raise ValueError("This target needs at least two dimensions: a1, x1")

        if self.ndim == 2:
            self._npar = 1

        if self.ndim > 3:
            self._npar = 2

        self._nx = self.ndim - self._npar

    def __call__(self, xarr):
        a1 = xarr[-1]
        a2 = 1.0 if self._npar == 1 else xarr[-2]
        x1 = xarr[0]

        arg = x1 * a2 + np.sum(xarr[1 : self._nx]) + a1
        return np.cos(arg)

    def integral(
        self, xmin, xmax, a1=None, a2=None, verbose=False, exact=True, marginalized_vars=None
    ):
        if marginalized_vars is not None:
            raise NotImplementedError("For Cosndalpha")

        if a1 is None:
            a1 = self.xmax[-1]
        if a2 is None:
            a2 = self.xmax[-self._npar]

        # Take only the x-part of the integration limits
        xmin = xmin[: self._nx]
        xmax = xmax[: self._nx]
        ranges = list(zip(xmin, xmax))

        if self.ndim == 2:
            extra = [a1]
        elif self.ndim > 3:
            extra = [a2, a1]

        fun = lambda *x: self(list(x) + extra)

        return nquad(fun, ranges)

    @property
    def xmin(self):
        if self._npar == 1:
            extra = [0.0]
        elif self._npar == 2:
            extra = [-0.5, 0.0]

        return [0.0] * self._nx + extra

    @property
    def xmax(self):
        if self._npar == 1:
            extra = [5.0]
        elif self._npar == 2:
            extra = [0.5, 5.0]

        return [3.5] * self._nx + extra

    @property
    def nderivatives(self):
        return self._nx


class ToyTarget(Cosnd):
    """
    Simple target to easily demonstrate how the methodologies outlined in the paper can be used to produce
    alpha-dependent differential distributions.
    As well as error estimation.

    It's similar to the `cosnd` target without the integration of the last dimension

    The target function is:
        cos( \sum{x_{i}} + alpha )
    """

    override = True

    def build(self):
        if self.ndim < 2:
            raise ValueError("This target needs at least two dimensions: a1, x1")

        if len(self._parameters) == self.ndim + 1:
            raise ValueError(
                "This function doesn't accept a parameter for the last dimension (which is not integrated over)"
            )

        super().build()

        # Add parameters specific to this target
        self._npar = 1
        self._nx = self.ndim - self._npar

    def integral(self, xmin, xmax, a1=None, *kwargs):
        if a1 is None:
            a1 = self.xmax[-1]

        # Take only the x-part of the integration limits
        xmin = xmin[: self._nx]
        xmax = xmax[: self._nx]
        ranges = list(zip(xmin, xmax))

        fun = lambda *x: self(list(x) + [a1])

        return nquad(fun, ranges)

    @property
    def xmin(self):
        return [0.0] * self._nx + [0.0]

    @property
    def xmax(self):
        return [3.5] * self._nx + [5.0]

    @property
    def nderivatives(self):
        return self._nx

    @property
    def xgrid(self):
        amin = self.xmin[-1]
        amax = self.xmax[-1]
        xmin = np.array(self.xmin[: self._nx])
        xmax = np.array(self.xmax[: self._nx])

        npoints = int(1e2)
        nr = 20
        xx = np.random.rand(npoints, self._nx) * (xmax - xmin) + xmin
        aa = np.linspace(amin, amax, npoints // nr)
        cc = [*xx.T, np.tile(aa, nr)]
        return np.vstack(cc).T


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

    The limits in Q are normalized between 0 and 1, but the PDF actually takes
    an actual Q**2 in GeV^2
    """

    max_par = 0
    max_ndim = 2
    override = True

    _min_x = 1e-3
    _min_q = 25.0
    _max_x = 0.7
    _max_q = 125.0

    def build(self):
        if self.ndim < 2:
            raise ValueError("This target, uquark2d, needs 2 dimensions: x,q")

        nx = 120
        nq = 100
        x = np.concatenate(
            [
                np.logspace(np.log10(self._min_x), -1, 2 * nx // 3),
                np.linspace(0.1, self._max_x, nx // 3),
            ]
        )

        q = np.linspace(0, 1, nq)
        xx, qq = np.meshgrid(x, q)
        self._xgrid = np.column_stack([xx.ravel(), qq.ravel()])

        self._pdf = mkPDF("nnpdf40/0", dirname=Path(__file__).parent)

    def __call__(self, xarr):
        x = xarr[0]
        q = self._min_q + xarr[1] * (self._max_q - self._min_q)
        return self._pdf.py_xfxQ2(2, [x], [q**2]).numpy()

    def __repr__(self):
        return f"xu(x)"

    def integral(self, xmin, xmax, scaled_q=1.0, verbose=True, exact=True):
        q = scaled_q * (self._max_q - self._min_q) + self._min_q
        if verbose:
            print(f"Computing the integral for {q=}")

        if exact:
            # This is slower but exact, otherwise just approximate the integration
            fun = lambda x: self([x, scaled_q])
            return nquad(fun, [(xmin, xmax)])

        npoints = 2000

        # Take only the x
        if not isinstance(xmin, (float, int)):
            xmin = xmin[0]
            xmax = xmax[0]

        # First integrate at small x
        xgrids = []
        if xmin < 0.1:
            xgrids.append(np.logspace(np.log(xmin), -1, npoints // 2))
        if xmax > 0.1:
            xgrids.append(np.linspace(0.1, xmax, npoints // 2))
        xgrid = np.concatenate(xgrids)

        # Get weights
        sp = np.diff(xgrid, append=xgrid[-1], prepend=xgrid[0])
        weights = (sp[1:] + sp[:-1]) / 2.0

        q2grid = np.ones_like(xgrid) * q**2

        pdf_vals = self._pdf.py_xfxQ2(2, xgrid, q2grid)

        return np.sum(pdf_vals * weights), 0.0

    def dimension_name(self, d):
        if d == 0:
            return "x"
        if d == 1:
            return "q"

    def dimension_scale(self, d):
        if d == 0:
            return "log"
        return "linear"

    @property
    def xmin(self):
        return [self._min_x, 0]

    @property
    def xmax(self):
        return [self._max_x, 1]

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
    "cosndalpha": CosndAlpha,
    "toy": ToyTarget,
}
