"""Qibo interface of the integrator"""
from copy import deepcopy
import dataclasses
from functools import cached_property
from multiprocessing import Manager, Pool
import os
import time

import numpy as np
from qibo import gates, hamiltonians, models, set_backend

set_backend("numpy")

GEN_EIGENVAL = 0.5  # Eigenvalue for the parameter shift rule of rotations
SHIFT = np.pi / (4.0 * GEN_EIGENVAL)
DERIVATIVE = True
ALPHA = 0.92 # alpha for the x^{alpha} factor of the PDF


def _recursive_shifts(arrays, index=1, s=SHIFT):
    """Recursively generate all the necessary shifts of all parameters of interest
    keeping also track of the sign of the factor that needs to be applied to the evaluation of the PSR

    Arrays must be a list of lists of _UploadedParameter
    (containing in the first iteration of the recursion, just one element)
    index is the number of variables for which we want to generate the shifts
    This allows the user to derivate with only a subset of the dimensions
    """
    index -= 1

    new_lists = []
    for list_parameters in arrays:
        for i, parameter in enumerate(list_parameters):
            if parameter.dimension == index:
                # Create a copy of the list with this parameter updated
                par_p = parameter.shift(s)
                par_m = parameter.shift(-s)
                new_list_up = deepcopy(list_parameters)
                new_list_do = deepcopy(list_parameters)
                new_list_up[i] = par_p
                new_list_do[i] = par_m
                new_lists.append(new_list_up)
                new_lists.append(new_list_do)

    # If at this point `new_list` is empty, it means we are marginalizing
    # and maybe we are skipping some derivatives, in that case, "reset"
    if not new_lists:
        new_lists = arrays

    if index == 0:
        return new_lists

    return _recursive_shifts(new_lists, index=index, s=s)


@dataclasses.dataclass
class _UploadedParameter:
    """This class holds the information necessary to upload variables to a circuit"""

    x: float
    theta: float
    index: int  # The parameter index in the circuit
    dimension: int  # The dimension this parameter corresponds to
    s: float = 0.0
    is_log: bool = False

    def shift(self, s):
        return dataclasses.replace(self, s=s)

    def unshift(self):
        return dataclasses.replace(self, s=0.0)

    @property
    def y(self):
        if self.is_log:
            x = np.log(self.x)
        else:
            x = self.x
        return x * self.theta + self.s

    @property
    def factor(self):
        if self.s == 0.0:
            return 1.0
        ret = np.sign(self.s) * self.theta
        if self.is_log:
            ret /= self.x
        return ret

    def __repr__(self):
        return f"{self.y} ({self.factor}, idx: {self.index})"


class BaseVariationalObservable:
    """Generate a variational circuit following the example at
    https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-write-a-custom-variational-circuit-optimization

    In this case the inputs of the function are injected in the `ndim` first parameters.
    """

    def __init__(
        self,
        nqubits=3,
        nlayers=3,
        ndim=1,
        nshots=None,
        initial_state=None,
        verbose=True,
        nderivatives=None,
    ):
        self._ndim = ndim
        self._nqubits = nqubits
        self._nlayers = nlayers
        self._nshots = nshots
        self._circuit = None
        self._observable = None
        self._variational_params = []
        self._initial_state = initial_state
        if nderivatives is None:
            self.nderivatives = ndim  # By default, derive all dimensions
        else:
            self.nderivatives = nderivatives
        self.pid = os.getpid()  # Useful information when multiprocessing

        # Set the reuploading indexes
        self._reuploading_indexes = [[] for _ in range(ndim)]

        self.build_circuit()
        self.build_observable()

        # setting initial random parameters
        if self._circuit is None:
            raise ValueError("Circuit has not being built")

        # Note that the total number of parameters in the circuit is the variational paramters + scaling
        self._nparams = len(self._circuit.get_parameters()) + 1
        # fixing the seed for reproducibility
        np.random.seed(1234)
        self._variational_params = np.random.randn(self._nparams - 1)
        self._scaling = 1.0

        # Visualizing the model
        if verbose:
            self.print_model()

    # Definition of some properties that can only be known upon first usage
    @cached_property
    def _eigenfactor(self):
        return GEN_EIGENVAL**self.nderivatives

    @cached_property
    def _backend_exe(self):
        return self._observable.backend.execute_circuit

    @cached_property
    def nparams(self):
        return self._nparams

    def __repr__(self):
        return f"{self.__class__.__name__} (dims={self._ndim}, nqubits={self._nqubits}, nlayers={self._nlayers}) running in process id: {self.pid}"

    def build_circuit(self):
        """Build step of the circuit"""
        # In this basic variational observable each x will be updated at a different layers
        # therefore the number of layers needs to be at least equal to the number of dimensions
        if self._nlayers < self._ndim:
            raise ValueError("BaseVariationalObservable needs at least a layer per dimension")

        circuit = models.Circuit(self._nqubits)

        for i in range(self._nlayers):
            # In this circuit, the reuploading indexes
            if i < self._ndim:
                curr_idx = len(circuit.get_parameters())
                self._reuploading_indexes[i].append(curr_idx)

            circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
            circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
            circuit.add((gates.CZ(q, q + 1) for q in range(0, self._nqubits - 1, 2)))
            circuit.add((gates.RX(q, theta=0) for q in range(self._nqubits)))
            circuit.add((gates.CZ(q, q + 1) for q in range(1, self._nqubits - 2, 2)))
            circuit.add(gates.CZ(0, self._nqubits - 1))

        circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
        self._circuit = circuit

    def print_model(self):
        """Print a model of the circuit"""
        print(
            f"""{self}
Circuit drawing:\n
    {self._circuit.draw()}
Circuit summary:
    {self._circuit.summary()}
"""
        )

    def build_observable(self):
        """Build step of the observable"""
        m0 = (1 / self._nqubits) * hamiltonians.Z(self._nqubits).matrix
        ham = hamiltonians.Hamiltonian(self._nqubits, m0)
        self._observable = ham

    def _upload_parameters(self, xarr):
        """Receives an array of x and returns the "uploaded" rotations
        i.e., y = theta_x * x

        Return all uploaded parameters as a list of instances of _UploadedParameter
        """
        y = []
        for d, (x, idxs) in enumerate(zip(xarr, self._reuploading_indexes)):
            for idx in idxs:
                theta = self._variational_params[idx]
                y.append(_UploadedParameter(x, theta, idx, d))
        return y

    def execute_with_x(self, xarr):
        y = self._upload_parameters(xarr)
        return self._execute(y)

    def _execute(self, uploaded_parameters):
        """Obtain the value of the observable for the given set of uploaded parameters
        by evaluating the circuit in those and computing the expectation value of the observable
        """
        # Update the parameters for this run
        circ_parameters = deepcopy(self._variational_params)
        for parameter in uploaded_parameters:
            circ_parameters[parameter.index] = parameter.y
        # Set the parameters in the circuit
        self._circuit.set_parameters(circ_parameters)
        # exact simulation if nshots is None
        if self._nshots is None:
            state = self._backend_exe(
                circuit=self._circuit, initial_state=self._initial_state
            ).state()
            expectation_value = self._observable.expectation(state)
        # shot-noise simulation otherwise
        else:
            expectation_value = self._backend_exe(
                circuit=self._circuit, initial_state=self._initial_state, nshots=self._nshots
            ).expectation_from_samples(self._observable)
            expectation_value = np.real(expectation_value)
        return expectation_value * self._scaling

    @property
    def parameters(self):
        return np.concatenate([[self._scaling], self._variational_params])

    def set_parameters(self, new_parameters):
        """Save the new set of parameters for the circuit
        They only get burned into the circuit when the forward pass is called
        The parameters must enter as a 1d array
        """
        self._scaling = new_parameters[0]
        self._variational_params = new_parameters[1:]

    def marginal_pass(self, xarr, nmarg):
        y = self._upload_parameters(xarr)

        shifts = _recursive_shifts([y], index=nmarg)

        res = 0.0
        for shift in shifts:
            factor = 1.0
            for val in shift:
                factor *= val.factor

            res += factor * self._execute(shift)

        return res * GEN_EIGENVAL**nmarg

    def forward_pass(self, xarr, derivate_ds=None):
        """Forward pass of the variational observable.
        This entails a parameter shift rule shift around the parameters xarr
        The number of dimension to derivate can be controlled via the nderivatives parameter
        the value fo the "nderivatives" property will be used
        """
        y = self._upload_parameters(xarr)
        eigenfactor = self._eigenfactor

        if derivate_ds is not None:
            # Remove dimension information from the variables we don't want to derivate
            # i.e., all except the marginalized
            for var in y:
                if (var.dimension + 1) not in derivate_ds:
                    var.dimension = -99

            eigenfactor = GEN_EIGENVAL ** len(derivate_ds)

        if DERIVATIVE:
            shifts = _recursive_shifts([y], index=self.nderivatives)
        else:
            shifts = [y]

        res = 0.0
        for shift in shifts:
            factor = 1.0
            for val in shift:
                factor *= val.factor

            res += factor * self._execute(shift)

        return eigenfactor * res


class ReuploadingAnsatz(BaseVariationalObservable):
    """Generates a variational quantum circuit in which we upload all the variables
    in each layer."""

    def __init__(self, nqubits, nlayers, ndim=1, **kwargs):
        """In this specific model the number of qubits is equal to the dimensionality
        of the problem."""
        if nqubits != ndim:
            raise ValueError(
                "For ReuploadingAnsatz the number of qubits must be equal to the number of dimensions"
            )
        # inheriting the BaseModel features
        super().__init__(nqubits, nlayers, ndim=ndim, **kwargs)

    def build_circuit(self):
        """Builds the reuploading ansatz for the circuit"""

        circuit = models.Circuit(self._nqubits)

        # At first we build up superposition for each qubit
        circuit.add((gates.H(q) for q in range(self._nqubits)))
        # then we add parametric gates
        for _ in range(self._nlayers):
            for q in range(self._nqubits):
                circuit.add(gates.RY(q, theta=0))
                self._reuploading_indexes[q].append(len(circuit.get_parameters()) - 1)
                circuit.add(gates.RY(q, theta=0))
            # if nqubits > 1 we build entanglement
            if self._nqubits > 1:
                circuit.add((gates.CZ(q, q + 1) for q in range(0, self._nqubits - 1, 1)))
                if self._nqubits > 2:
                    circuit.add((gates.CZ(self._nqubits - 1, 0)))
        # final rotation only with more than 1 qubit
        if self._nqubits > 1:
            circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
        # measurement gates
        circuit.add((gates.M(q) for q in range(self._nqubits)))

        self._circuit = circuit

        # Get the initial parameters
        self._variational_params = np.array(circuit.get_parameters()).flatten()


class DeepReuploading(BaseVariationalObservable):
    """
    This ansatz is in principle equivalent to ReuploadingAnsatz, but in this case
    we implement the full Fourier layer purposed in: https://arxiv.org/abs/1907.02085.
    """

    def build_circuit(self):
        """Builds the reuploading ansatz for the circuit"""

        circuit = models.Circuit(self._nqubits)

        # At first we build up superposition for each qubit
        circuit.add((gates.H(q) for q in range(self._nqubits)))
        # then we add parametric gates
        for _ in range(self._nlayers):
            for q in range(self._nqubits):
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
                self._reuploading_indexes[q].append(len(circuit.get_parameters()) - 1)
                circuit.add(gates.RZ(q, theta=0))
                circuit.add(gates.RY(q, theta=0))
                circuit.add(gates.RZ(q, theta=0))
            # if nqubits > 1 we build entanglement
            if self._nqubits > 1:
                circuit.add((gates.CZ(q, q + 1) for q in range(0, self._nqubits - 1, 1)))
                if self._nqubits > 2:
                    circuit.add((gates.CZ(self._nqubits - 1, 0)))
        # final rotation only with more than 1 qubit
        if self._nqubits > 1:
            circuit.add((gates.RY(q, theta=0) for q in range(self._nqubits)))
        # measurement gates
        circuit.add((gates.M(q) for q in range(self._nqubits)))

        self._circuit = circuit

        # Get the initial parameters
        self._variational_params = np.array(circuit.get_parameters()).flatten()


class VerticalUploading(BaseVariationalObservable):
    """
    Builds a vertical reuploading strategy.
    With this ansatz each feature is uploaded in each qubit and each layer
    following the Pérez-Salinas anzats: https://arxiv.org/abs/1907.02085.
    """

    def build_sheet(self, circuit, q, x_idx):
        """
        Uploading layer for one variable corresponding to index x_idx.
        """
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        self._reuploading_indexes[x_idx].append(len(circuit.get_parameters()) - 1)
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def build_circuit(self):
        """Builds the reuploading ansatz for the circuit"""

        circuit = models.Circuit(self._nqubits)

        for _ in range(self._nlayers):
            for q in range(self._nqubits):
                for dim in range(self._ndim):
                    self.build_sheet(circuit, q, dim)
            # if nqubits > 1 we build entanglement
            if self._nqubits > 1:
                circuit.add((gates.CZ(q, q + 1) for q in range(0, self._nqubits - 1, 1)))
                if self._nqubits > 2:
                    circuit.add((gates.CZ(self._nqubits - 1, 0)))

        # measurement gates
        circuit.add((gates.M(q) for q in range(self._nqubits)))

        self._circuit = circuit

        # Get the initial parameters
        self._variational_params = np.array(circuit.get_parameters()).flatten()


class qPDFAnsatz(BaseVariationalObservable):
    """
    Generates a circuit which follows the qPDF ansatz.
    The following implementation works only for 1 flavour; uquark in this case.
    Ref: https://arxiv.org/abs/2011.13934.

    It accepts two input variables (two dimensions) but only the first one will have the logarithm upload
    """

    def __init__(self, nqubits, nlayers, ndim=1, **kwargs):
        """In this specific model we are going to use a 1 qubit circuit."""
        if ndim > 2:
            raise ValueError("Only 2 dimensions can be fitted with qPDF")
        self._logarithm_variables = []
        super().__init__(nqubits, nlayers, ndim=ndim, **kwargs)
        self.nderivatives = 1  # The derivative is performed only on the first dimension

    def _upload_parameters(self, xarr):
        yarr = super()._upload_parameters(xarr)
        for y in yarr:
            if y.index in self._logarithm_variables:
                y.is_log = True
        return yarr

    def build_circuit(self):
        """Builds the reuploading ansatz for the circuit"""

        circuit = models.Circuit(self._nqubits)

        # then we add parametric gates
        for i in range(self._nlayers):
            circuit.add(gates.RY(q=0, theta=0))
            idx = len(circuit.get_parameters()) - 1
            self._reuploading_indexes[0].append(idx)
            self._logarithm_variables.append(idx)
            circuit.add(gates.RY(q=0, theta=0))

            if self._ndim > 1:
                # Add a gate for the second dimension
                circuit.add(gates.RY(q=0, theta=0))
                idx = len(circuit.get_parameters()) - 1
                self._reuploading_indexes[1].append(idx)
                self._logarithm_variables.append(idx)
                circuit.add(gates.RY(q=0, theta=0))

            if i != (self._nlayers - 1):
                circuit.add(gates.RZ(q=0, theta=0))
                self._reuploading_indexes[0].append(len(circuit.get_parameters()) - 1)
                circuit.add(gates.RZ(q=0, theta=0))

        # measurement gates
        circuit.add((gates.M(0)))

        self._circuit = circuit
        # Get the initial parameters
        self._variational_params = np.array(circuit.get_parameters()).flatten()

    def execute_with_x(self, xarr):
        ret = super().execute_with_x(xarr)
        return xarr[0] ** ALPHA * ret

    def forward_pass(self, xarr):
        circ = super().execute_with_x(xarr)
        der = super().forward_pass(xarr)
        return xarr[0] ** ALPHA * der + ALPHA * circ * xarr[0] ** (ALPHA - 1.0)


class qPDF_v2(qPDFAnsatz):
    """Alternative version for the qPDF problem."""

    def __init__(self, nqubits, nlayers, ndim=1, **kwargs):
        """In this specific model we are going to use a 2 qubit circuit."""
        if nqubits > 1 or ndim > 2:
            raise ValueError(
                "With this ansatz we tackle the 1d uquark qPDF and with a 1 qubits model."
            )
        # inheriting the BaseModel features
        super().__init__(nqubits, nlayers, ndim=ndim, **kwargs)

    def build_circuit(self):
        """Builds the reuploading ansatz for the circuit"""

        circuit = models.Circuit(self._nqubits)

        for i in range(self._nlayers):
            if i < (self._nlayers - 1):
                # Add the q-rotation at the beginning of the layers
                # and not for the last one
                # q
                circuit.add(gates.RZ(q=0, theta=0))
                idx = len(circuit.get_parameters()) - 1
                self._reuploading_indexes[1].append(idx)
                # bias on q
                circuit.add(gates.RZ(q=0, theta=0))

            # first x upload: log(x)
            circuit.add(gates.RY(q=0, theta=0))
            idx = len(circuit.get_parameters()) - 1
            self._reuploading_indexes[0].append(idx)
            self._logarithm_variables.append(idx)
            # bias on log(x)
            circuit.add(gates.RY(q=0, theta=0))

            # second x upload: (x)
            circuit.add(gates.RZ(q=0, theta=0))
            self._reuploading_indexes[0].append(len(circuit.get_parameters()) - 1)
            circuit.add(gates.RZ(q=0, theta=0))

        # measurement gates
        circuit.add((gates.M(*range(self._nqubits))))

        self._circuit = circuit
        # Get the initial parameters
        self._variational_params = np.array(circuit.get_parameters()).flatten()


class GoodScaling(BaseVariationalObservable):
    """In this ansatz we upload 2 variables per qubit."""

    def __init__(self, nqubits, nlayers, ndim=1, **kwargs):
        """In this specific model we are going to upload 2 variables to each qubit."""
        if nqubits != (int((ndim - 1) / 2) + 1):
            raise ValueError("Please set the number of qubits to be equal to int(ndim/2)+1.")
        # inheriting the BaseModel features
        super().__init__(nqubits, nlayers, ndim=ndim, **kwargs)

    def build_sheet(self, circuit, x_idx):
        """Uploading layer for a target variable x_idx"""
        q = int(x_idx / 2)

        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))
        self._reuploading_indexes[x_idx].append(len(circuit.get_parameters()) - 1)
        circuit.add(gates.RZ(q, theta=0))
        circuit.add(gates.RY(q, theta=0))
        circuit.add(gates.RZ(q, theta=0))

    def build_circuit(self):
        """Builds the reuploading ansatz for the circuit"""

        circuit = models.Circuit(self._nqubits)

        for _ in range(self._nlayers):
            for dim in range(self._ndim):
                self.build_sheet(circuit, dim)
            # if nqubits > 1 we build entanglement
            if self._nqubits > 1:
                circuit.add((gates.CZ(q, q + 1) for q in range(0, self._nqubits - 1, 1)))
                if self._nqubits > 2:
                    circuit.add((gates.CZ(self._nqubits - 1, 0)))
        for q in range(self._nqubits):
            circuit.add(gates.RY(q, theta=0))

        # measurement gates
        circuit.add((gates.M(q) for q in range(self._nqubits)))

        self._circuit = circuit

        # Get the initial parameters
        self._variational_params = np.array(circuit.get_parameters()).flatten()


available_ansatz = {
    "base": BaseVariationalObservable,
    "reuploading": ReuploadingAnsatz,
    "deepup": DeepReuploading,
    "verticup": VerticalUploading,
    "qpdf": qPDFAnsatz,
    "qpdf2q": qPDF_v2,
    "goodscaling": GoodScaling,
}


#### Pooling management
def initialize_pool(observable_class, nqubits, nlayers, ndim, nshots, nderivatives):
    global my_ansatz
    my_ansatz = observable_class(
        nqubits=nqubits,
        nlayers=nlayers,
        ndim=ndim,
        nshots=nshots,
        verbose=False,
        nderivatives=nderivatives,
    )
    return my_ansatz


def worker_set_parameters(parameters, running_pool):
    my_ansatz.set_parameters(parameters)

    # Remove one element of running_pool and wait
    _ = running_pool.pop()
    while running_pool:
        time.sleep(1e-6)
    return my_ansatz.pid


def worker_forward_pass(xarr):
    return my_ansatz.forward_pass(xarr)


def worker_get_ansatz(_):
    return my_ansatz


class ObservablePool:
    def __init__(self, pool):
        self._pool = pool
        self._nprocesses = pool._processes
        self._ansatz = pool.map(worker_get_ansatz, [None])[0]
        self._ansatz.print_model()
        self._ret = []
        if self._nprocesses > 1:
            self._manager = Manager()
            self._shr = self._manager.list()

    # Pooled calls
    def set_parameters(self, parameters):
        parameters = parameters.reshape(-1)

        # Make sure that the main process ansatz has the right parameters
        self._ansatz.set_parameters(parameters)
        if self._nprocesses == 1:
            return

        # Now prepare a list with locks
        for _ in range(self._nprocesses):
            self._shr.append(None)

        # Send the parameter set as async calls
        pids = [
            self._pool.apply_async(worker_set_parameters, args=(parameters, self._shr))
            for _ in range(self._nprocesses)
        ]

        # And wait until all threads are done before continuing
        [i.get() for i in pids]

    def vectorized_forward_pass(self, all_xarr):
        if self._nprocesses == 1:
            return np.array([self.forward_pass(i) for i in all_xarr])
        return np.array(self._pool.map(worker_forward_pass, all_xarr))

    # The rest are passed silently to the original ansatz
    @cached_property
    def nderivatives(self):
        return self._ansatz.nderivatives

    @property
    def parameters(self):
        return self._ansatz.parameters

    def forward_pass(self, xarr):
        return self._ansatz.forward_pass(xarr)

    def execute_with_x(self, xarr):
        return self._ansatz.execute_with_x(xarr)

    def __del__(self):
        """Liberate processes and memory"""
        if self._nprocesses > 1:
            del self._shr
            self._manager.shutdown()
            self._pool.terminate()


class FakePool:
    _processes = 1

    def __init__(self, initargs=(), **kwargs):
        self._ansatz = initialize_pool(*initargs)

    def map(self, function, *args):
        if function.__name__ == "worker_get_ansatz":
            return [self._ansatz]


def generate_ansatz_pool(
    observable_class, nshots=None, nqubits=1, nlayers=1, ndim=1, nprocesses=1, nderivatives=None
):
    """Generate a pool of ansatz"""
    if nprocesses == 1:
        pool_class = FakePool
    else:
        pool_class = Pool

    pool = pool_class(
        processes=nprocesses,
        initializer=initialize_pool,
        initargs=(observable_class, nqubits, nlayers, ndim, nshots, nderivatives),
    )

    return ObservablePool(pool)
