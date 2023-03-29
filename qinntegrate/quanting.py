"""Qibo interface of the integrator"""
import dataclasses
from copy import deepcopy
import numpy as np
from qibo import models, gates, hamiltonians, set_backend

set_backend("numpy")

GEN_EIGENVAL = 0.5  # Eigenvalue for the parameter shift rule of rotations
SHIFT = np.pi / (4.0 * GEN_EIGENVAL)


def _recursive_shifts(arrays, index=1, s=SHIFT):
    """Recursively generate all the necessary shifts of all parameters of interest
    keeping also track of the sign of the factor that needs to be applied to the evaluation of the PSR

    Arrays must be a list of lists of _UploadedParameter
    (containing in the first iteration of the recursion, just one element)
    index is the number of variable for which we want to generate the shifts
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

    def shift(self, s):
        return dataclasses.replace(self, s=s)

    def unshift(self):
        return dataclasses.replace(self, s=0.0)

    @property
    def y(self):
        return self.x * self.theta + self.s

    @property
    def factor(self):
        # The shift-factor is only active once per dimension
        if self.s == 0.0:
            return 1.0
        return np.sign(self.s) * self.theta

    def __repr__(self):
        return f"{self.y} ({self.factor}, idx: {self.index})"


class BaseVariationalObservable:
    """Generate a variational circuit following the example at
    https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-write-a-custom-variational-circuit-optimization

    In this case the inputs of the function are injected in the `ndim` first parameters.
    """

    def __init__(self, nqubits=3, nlayers=3, ndim=1, initial_state=None):
        self._ndim = ndim
        self._nqubits = nqubits
        self._nlayers = nlayers
        self._circuit = None
        self._observable = None
        self._variational_params = []
        self._initial_state = initial_state
        self._eigenfactor = GEN_EIGENVAL**ndim

        # Set the reuploading indexes
        self._reuploading_indexes = [[] for _ in range(ndim)]

        self.build_circuit()
        self.build_observable()

        # setting initial random parameters
        if self._circuit is None:
            raise ValueError("Circuit has not being built")
        self._nparams = len(self._circuit.get_parameters())
        self._variational_params = np.random.randn(self._nparams)

        # Visualizing the model
        self.print_model()

    def __repr__(self):
        return self.__class__.__name__

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
        print(f"\nCircuit drawing:\n{self._circuit.draw()}\n")
        print(f"Circuit summary:\n{self._circuit.summary()}\n")

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
                theta = self.parameters[idx]
                y.append(_UploadedParameter(x, theta, idx, d))
        return y

    def execute_with_x(self, xarr):
        y = self._upload_parameters(xarr)
        return self._execute(y)

    def _execute(self, uploaded_paramaters):
        """Obtain the value of the observable for the given set of uploaded parameters
        by evaluating the circuit in those and computing the expectation value of the observable
        """
        bexec = self._observable.backend.execute_circuit  # is this needed?
        # Update the parameters for this run
        circ_parameters = deepcopy(self.parameters)
        for i, parameter in enumerate(uploaded_paramaters):
            if str(self._observable.backend) == "tensorflow":
                circ_parameters[i].assign(parameter.y)
            else:
                circ_parameters[i] = parameter.y
        self._circuit.set_parameters(circ_parameters)
        state = bexec(circuit=self._circuit, initial_state=self._initial_state).state()
        return self._observable.expectation(state)

    @property
    def parameters(self):
        return self._variational_params

    @property
    def nparams(self):
        return len(self._variational_params)

    def set_parameters(self, new_parameters):
        """Save the new set of parameters for the circuit
        They only get burned into the circuit when the forward pass is called
        """

        if new_parameters.shape[0] != self.nparams:
            raise ValueError("Trying to set more parameters than those allowed by the circuit")
        self._variational_params = new_parameters

    def forward_pass(self, xarr):
        """Forward pass of the variational observable.
        This entails a parameter shift rule shift around the parameters xarr
        """
        y = self._upload_parameters(xarr)

        shifts = _recursive_shifts([y], index=self._ndim)

        res = 0.0
        for shift in shifts:
            factor = 1.0
            for val in shift:
                factor *= val.factor

            res += factor * self._execute(shift)

        return self._eigenfactor * res



class ReuploadingAnsatz(BaseVariationalObservable):
    """Generates a variational quantum circuit in which we upload all the variables
    in each layer."""

    def __init__(self, nqubits, nlayers, ndim=1, initial_state=None):
        """In this specific model the number of qubits is equal to the dimensionality
        of the problem."""
        if nqubits != ndim:
            raise ValueError(
                "For ReuploadingAnsatz the number of qubits must be equal to the number of dimensions"
            )
        # inheriting the BaseModel features
        super().__init__(nqubits, nlayers, ndim=ndim, initial_state=initial_state)

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


class qPDFAnsatz(BaseVariationalObservable):
    """
    Generates a circuit which follows the qPDF ansatz.
    The following implementation works only for 1 flavour; uquark in this case.
    Ref: https://arxiv.org/abs/2011.13934.
    """

    def __init__(self, nqubits, nlayers, ndim=1, initial_state=None):
        """In this specific model we are going to use a 1 qubit circuit."""
        if nqubits != 1 or ndim != 1:
            raise ValueError(
                "With this ansatz we tackle the 1d uquark qPDF and only 1 qubit is allowed."
            )
        # inheriting the BaseModel features
        super().__init__(nqubits, nlayers, ndim=ndim, initial_state=initial_state)

    def build_circuit(self):
        """Builds the reuploading ansatz for the circuit"""

        circuit = models.Circuit(self._nqubits)

        # then we add parametric gates
        for _ in range(self._nlayers):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # THIS ROTATION MUST BE FILLED WITH: log(x)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            circuit.add(gates.RZ(q=0, theta=0))
            self._reuploading_indexes[0].append(len(circuit.get_parameters()) - 1)
            circuit.add(gates.RZ(q=0, theta=0))
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # THIS ROTATION MUST BE FILLED WITH: x
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            circuit.add(gates.RY(q=0, theta=0))
            self._reuploading_indexes[0].append(len(circuit.get_parameters()) - 1)
            circuit.add(gates.RY(q=0, theta=0))
        # measurement gates
        circuit.add((gates.M(0)))

        self._circuit = circuit
        # Get the initial parameters
        self._variational_params = np.array(circuit.get_parameters()).flatten()


available_ansatz = {"base": BaseVariationalObservable, "reuploading": ReuploadingAnsatz, "qpdf" : qPDFAnsatz}